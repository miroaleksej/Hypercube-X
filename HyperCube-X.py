import numpy as np
import torch
import gpytorch
import networkx as nx
import logging
import json
import zstandard as zstd
import time
from collections import deque
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from giotto_tda.homology import VietorisRipsPersistence
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms import VQC
from qiskit.quantum_info import state_fidelity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import psutil
import GPUtil
from threading import Lock, Thread
from functools import lru_cache
import hashlib
from collections import OrderedDict
import random
import pickle
import zlib
import datetime
from logging.handlers import RotatingFileHandler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from giotto_tda.diagrams import BettiCurve
import torch.nn as nn
from torch.autograd import Function
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, partial_trace
from typing import Dict, List, Tuple, Optional, Union
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

# ===================================================================
# Класс GPUComputeManager
# ===================================================================
class GPUComputeManager:
    def __init__(self, resource_threshold=0.8):
        self.resource_threshold = resource_threshold
        self.compute_lock = Lock()
        self.gpu_available = self._detect_gpu_capability()
        self.logger = logging.getLogger("GPUManager")
        self.last_utilization = {'cpu': 0.0, 'gpu': 0.0}
        
    def _detect_gpu_capability(self):
        """Обнаружение доступных GPU с проверкой памяти"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                self.logger.info("No GPUs detected")
                return False
                
            for i, gpu in enumerate(gpus):
                self.logger.info(f"GPU {i}: {gpu.name}, Free: {gpu.memoryFree}MB, Total: {gpu.memoryTotal}MB")
            
            return True
        except Exception as e:
            self.logger.error(f"GPU detection failed: {str(e)}")
            return False
    
    def _get_gpu_status(self):
        """Получение текущей загрузки GPU"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0.0
                
            return max(gpu.memoryUtil for gpu in gpus)
        except:
            return 0.0
    
    def _get_cpu_status(self):
        """Получение текущей загрузки CPU"""
        return psutil.cpu_percent() / 100.0
    
    def _check_resources(self):
        """Проверка загрузки системы"""
        cpu_load = self._get_cpu_status()
        gpu_load = self._get_gpu_status()
        self.last_utilization = {'cpu': cpu_load, 'gpu': gpu_load}
        return cpu_load < self.resource_threshold and gpu_load < self.resource_threshold
    
    def get_resource_utilization(self):
        """Возвращает текущую загрузку ресурсов"""
        return self.last_utilization
    
    def execute(self, func, *args, **kwargs):
        """
        Выполнение функции с контролем ресурсов
        Возвращает результат вычислений и флаг использования GPU
        """
        with self.compute_lock:
            start_wait = time.time()
            while not self._check_resources():
                time.sleep(0.1)
                if time.time() - start_wait > 30:
                    self.logger.warning("Resource wait timeout exceeded")
                    break
            
            if self.gpu_available and torch.cuda.is_available():
                device = torch.device("cuda")
                backend = "cuda"
            else:
                device = torch.device("cpu")
                backend = "cpu"
            
            start_time = time.time()
            result = func(device, *args, **kwargs)
            compute_time = time.time() - start_time
            
            self.logger.info(f"Computation completed with {backend} in {compute_time:.4f}s")
            return result

# ===================================================================
# Класс SmartCache
# ===================================================================
class SmartCache:
    def __init__(self, max_size=10000, ttl_minutes=30, cache_dir="cache"):
        self.max_size = max_size
        self.ttl = datetime.timedelta(minutes=ttl_minutes)
        self.cache_dir = cache_dir
        self.memory_cache = OrderedDict()
        self.logger = logging.getLogger("SmartCache")
        self.eviction_lock = Lock()
        
        os.makedirs(cache_dir, exist_ok=True)
        self.cleanup_thread = Thread(target=self._periodic_cleanup, daemon=True)
        self.cleanup_thread.start()
    
    def _key_to_hash(self, key):
        """Преобразование ключа в хеш"""
        if isinstance(key, (list, dict, np.ndarray)):
            key = json.dumps(key, sort_keys=True, default=str)
        return hashlib.sha256(str(key).encode()).hexdigest()
    
    def _get_cache_path(self, key):
        """Получение пути к файлу кэша"""
        return os.path.join(self.cache_dir, f"{key}.cache")
    
    def _is_expired(self, timestamp):
        """Проверка истечения срока действия"""
        return datetime.datetime.now() - timestamp > self.ttl
    
    def _periodic_cleanup(self):
        """Периодическая очистка кэша"""
        while True:
            time.sleep(300)
            self.clear_expired()
    
    def set(self, key, value, is_permanent=False):
        """Установка значения в кэш"""
        key_hash = self._key_to_hash(key)
        now = datetime.datetime.now()
        
        with self.eviction_lock:
            self.memory_cache[key_hash] = {
                "value": value,
                "timestamp": now,
                "is_permanent": is_permanent
            }
            self.memory_cache.move_to_end(key_hash)
            
            if len(self.memory_cache) > self.max_size:
                self._evict_cache()
        
        cache_path = self._get_cache_path(key_hash)
        cache_data = {
            "value": value,
            "timestamp": now.isoformat(),
            "is_permanent": is_permanent
        }
        
        compressed = zlib.compress(pickle.dumps(cache_data))
        with open(cache_path, "wb") as f:
            f.write(compressed)
    
    def get(self, key):
        """Получение значения из кэш"""
        key_hash = self._key_to_hash(key)
        now = datetime.datetime.now()
        
        with self.eviction_lock:
            if key_hash in self.memory_cache:
                entry = self.memory_cache[key_hash]
                if entry["is_permanent"] or not self._is_expired(entry["timestamp"]):
                    self.memory_cache.move_to_end(key_hash)
                    return entry["value"]
                else:
                    del self.memory_cache[key_hash]
        
        cache_path = self._get_cache_path(key_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    compressed = f.read()
                cache_data = pickle.loads(zlib.decompress(compressed))
                timestamp = datetime.datetime.fromisoformat(cache_data["timestamp"])
                
                if cache_data["is_permanent"] or not self._is_expired(timestamp):
                    with self.eviction_lock:
                        self.memory_cache[key_hash] = {
                            "value": cache_data["value"],
                            "timestamp": timestamp,
                            "is_permanent": cache_data["is_permanent"]
                        }
                    return cache_data["value"]
                else:
                    os.remove(cache_path)
            except Exception as e:
                self.logger.error(f"Cache read error: {str(e)}")
        
        return None
    
    def _evict_cache(self):
        """Вытеснение наименее используемых записей"""
        temp_entries = [k for k, v in self.memory_cache.items() if not v["is_permanent"]]
        
        if not temp_entries:
            return
            
        temp_entries.sort(key=lambda k: self.memory_cache[k]["timestamp"])
        
        eviction_count = max(1, len(temp_entries)//10)
        for key in temp_entries[:eviction_count]:
            del self.memory_cache[key]
    
    def clear_expired(self):
        """Очистка просроченных записей"""
        now = datetime.datetime.now()
        expired_keys = []
        
        with self.eviction_lock:
            for key, entry in self.memory_cache.items():
                if not entry["is_permanent"] and self._is_expired(entry["timestamp"]):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".cache"):
                try:
                    filepath = os.path.join(self.cache_dir, filename)
                    with open(filepath, "rb") as f:
                        compressed = f.read()
                    cache_data = pickle.loads(zlib.decompress(compressed))
                    
                    if not cache_data.get("is_permanent", False):
                        timestamp = datetime.datetime.fromisoformat(cache_data["timestamp"])
                        if self._is_expired(timestamp):
                            os.remove(filepath)
                except:
                    pass

# ===================================================================
# Класс ExactGPModel
# ===================================================================
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel="RBF", physical_constraints=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.physical_constraints = physical_constraints or []
        
        if kernel == "Matern":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel()
            )
        elif kernel == "RationalQuadratic":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RQKernel()
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
    
    def apply_constraints(self, x, mean_x, covar_x):
        return mean_x, covar_x
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ===================================================================
# Класс TopologicalQuantumCore (Hypercube-X)
# ===================================================================
class TopologicalQuantumCore:
    """Квантовое ядро для топологических вычислений"""
    def __init__(self, n_qubits=8):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend('statevector_simulator')
        self.logger = logging.getLogger("TopologicalQuantumCore")
        
    def create_entanglement_circuit(self, topology_data):
        """Создает квантовую схему, отражающую топологию системы"""
        circuit = QuantumCircuit(self.n_qubits)
        
        for i, feature in enumerate(topology_data[:self.n_qubits]):
            circuit.rx(feature, i)
            
        for i in range(self.n_qubits-1):
            circuit.cx(i, i+1)
            
        return circuit
    
    def calculate_topological_fidelity(self, circuit1, circuit2):
        """Вычисляет топологическую верность между двумя состояниями"""
        state1 = self.execute_circuit(circuit1)
        state2 = self.execute_circuit(circuit2)
        return state_fidelity(state1, state2)
    
    def execute_circuit(self, circuit):
        """Выполняет квантовую схему и возвращает состояние"""
        result = execute(circuit, self.backend).result()
        return result.get_statevector()

# ===================================================================
# Класс TopologyDynamicsEngine (Hypercube-X)
# ===================================================================
class TopologyDynamicsEngine:
    """Двигатель эволюции топологии"""
    def __init__(self, hypercube_system):
        self.system = hypercube_system
        self.topology_history = deque(maxlen=10)
        self.quantum_core = TopologicalQuantumCore()
        self.current_circuit = None
        self.logger = logging.getLogger("TopologyDynamicsEngine")
        
    def initialize_topology(self):
        """Инициализация начальной топологии"""
        self.calculate_topology()
        self.current_circuit = self.quantum_core.create_entanglement_circuit(
            self.get_topology_vector()
        )
        self.topology_history.append(self.get_topology_snapshot())
        
    def get_topology_vector(self):
        """Возвращает векторное представление топологии"""
        betti = self.system.topological_invariants.get('betti_numbers', {})
        return [
            betti.get(0, 0),
            betti.get(1, 0),
            betti.get(2, 0),
            len(self.system.critical_points),
            len(self.system.symmetries)
        ]
    
    def get_topology_snapshot(self):
        """Создает снимок текущей топологии"""
        return {
            'betti_numbers': self.system.topological_invariants.get('betti_numbers', {}).copy(),
            'critical_points': [cp.copy() for cp in self.system.critical_points],
            'symmetries': self.system.symmetries.copy(),
            'quantum_circuit': self.current_circuit.copy() if self.current_circuit else None
        }
    
    def calculate_topology(self):
        """Пересчитывает топологические инварианты"""
        sample_size = max(10, len(self.system.known_points) // 5)
        sample_indices = np.random.choice(len(self.system.known_points), sample_size, replace=False)
        X = np.array([self.system.known_points[i] for i in sample_indices])
        
        homology_dimensions = [0, 1, 2]
        vr = VietorisRipsPersistence(homology_dimensions=homology_dimensions)
        diagrams = vr.fit_transform([X])
        
        self.system.topological_invariants = {
            'persistence_diagrams': diagrams,
            'betti_numbers': {
                dim: int(np.sum(BettiCurve().fit_transform(diagrams)[0][:, dim] > 0.1))
                for dim in homology_dimensions
            }
        }
        
    def evaluate_topology_change(self, new_point, new_value):
        """Оценивает влияние новой точки на топологию"""
        if self.is_anomaly(new_point, new_value):
            return True
        
        if self.breaks_symmetry(new_point, new_value):
            return True
        
        new_vector = self.get_topology_vector()
        new_circuit = self.quantum_core.create_entanglement_circuit(new_vector)
        fidelity = self.quantum_core.calculate_topological_fidelity(
            self.current_circuit, new_circuit
        )
        
        return fidelity < 0.7
    
    def is_anomaly(self, point, value):
        """Определяет, является ли точка аномалией"""
        mean_val = np.mean(self.system.known_values)
        std_val = np.std(self.system.known_values)
        return abs(value - mean_val) > 5 * std_val
    
    def breaks_symmetry(self, point, value):
        """Проверяет, нарушает ли точка существующие симметрии"""
        for dim, sym_data in self.system.symmetries.items():
            if sym_data['type'] == 'reflection':
                idx = self.system.dim_names.index(dim)
                center = sym_data['center']
                symmetric_point = point.copy()
                symmetric_point[idx] = 2*center - point[idx]
                symmetric_value = self.system.physical_query_dict(
                    {dim: symmetric_point[i] for i, dim in enumerate(self.system.dim_names)}
                )
                if abs(value - symmetric_value) > 0.1 * abs(value):
                    return True
        return False
    
    def evolve_topology(self):
        """Выполняет эволюцию топологии системы"""
        self.calculate_topology()
        new_vector = self.get_topology_vector()
        new_circuit = self.quantum_core.create_entanglement_circuit(new_vector)
        
        self.current_circuit = new_circuit
        self.topology_history.append(self.get_topology_snapshot())
        self.logger.info("Topology evolved successfully")
        
        for child in self.system.child_hypercubes:
            child.topology_engine.evolve_topology()

# ===================================================================
# Класс TopologicalEnsembleInterface (Hypercube-X)
# ===================================================================
class TopologicalEnsembleInterface:
    """Интерфейс для работы с ансамблем систем"""
    def __init__(self, base_hypercube):
        self.base = base_hypercube
        self.parallel_systems = {}
        self.logger = logging.getLogger("TopologicalEnsembleInterface")
    
    def create_parallel_system(self, system_id, modification_rules):
        """Создает параллельную систему с модифицированными законами"""
        new_system = DynamicPhysicsHypercube(
            self.base.dimensions.copy(),
            resolution=self.base.resolution
        )
        
        for dim, rule in modification_rules.items():
            if rule['type'] == 'shift':
                min_val, max_val = new_system.dimensions[dim]
                shift = rule['amount']
                new_system.dimensions[dim] = (min_val + shift, max_val + shift)
            elif rule['type'] == 'scale':
                min_val, max_val = new_system.dimensions[dim]
                center = (min_val + max_val) / 2
                new_min = center + (min_val - center) * rule['factor']
                new_max = center + (max_val - center) * rule['factor']
                new_system.dimensions[dim] = (new_min, new_max)
        
        self.parallel_systems[system_id] = new_system
        self.logger.info(f"Created parallel system '{system_id}'")
        return new_system
    
    def compare_systems(self, system_id1, system_id2):
        """Сравнивает две параллельные системы"""
        sys1 = self.parallel_systems[system_id1]
        sys2 = self.parallel_systems[system_id2]
        
        betti1 = sys1.topological_invariants.get('betti_numbers', {})
        betti2 = sys2.topological_invariants.get('betti_numbers', {})
        
        coherence1 = self._measure_system_coherence(sys1)
        coherence2 = self._measure_system_coherence(sys2)
        
        return {
            'betti_difference': {k: betti1.get(k,0) - betti2.get(k,0) for k in set(betti1)|set(betti2)},
            'coherence_difference': coherence1 - coherence2,
            'stability_ratio': self._calculate_stability_ratio(sys1, sys2)
        }
    
    def _measure_system_coherence(self, system):
        """Измеряет квантовую когерентность системы"""
        if not system.quantum_model:
            return 0
        return len(system.critical_points) / (len(system.known_points) + 1e-5)
    
    def _calculate_stability_ratio(self, sys1, sys2):
        """Вычисляет относительную стабильность систем"""
        entropy1 = self._calculate_entropy_metrics(sys1)['shannon_entropy']
        entropy2 = self._calculate_entropy_metrics(sys2)['shannon_entropy']
        
        return entropy2 / (entropy1 + 1e-10)

# ===================================================================
# Класс TopologicalHypercubeOptimizer (Hypercube-X)
# ===================================================================
class TopologicalHypercubeOptimizer:
    def __init__(self, hypercube_system):
        """
        Инициализация оптимизатора с системой Hypercube-X
        :param hypercube_system: экземпляр DynamicPhysicsHypercube
        """
        self.system = hypercube_system
        self.logger = logging.getLogger("TopologicalHypercubeOptimizer")
        self.dimensionality_graph = nx.Graph()
        
        # Автонастройка параметров оптимизации
        self._auto_configure()
        self.logger.info("TopologicalHypercubeOptimizer initialized")

    def _auto_configure(self):
        """Автоматическая настройка параметров оптимизации"""
        num_points = len(self.system.known_points)
        if num_points < 50:
            self.symbolic_regression_generations = 20
            self.quantum_depth = 2
        elif num_points < 200:
            self.symbolic_regression_generations = 40
            self.quantum_depth = 3
        else:
            self.symbolic_regression_generations = 60
            self.quantum_depth = 4
            
        self.logger.info(f"Auto-configured: symbolic_generations={self.symbolic_regression_generations}, quantum_depth={self.quantum_depth}")

    def topological_dimensionality_reduction(self, target_dim=3):
        """
        Топологическая редукция размерности с сохранением гомотопических свойств
        с использованием динамической адаптации под текущую топологию
        """
        try:
            X = np.array(self.system.known_points)
            y = np.array(self.system.known_values)
            
            # Динамический выбор метода на основе топологии
            if self.system.topological_invariants.get('betti_numbers', {}).get(1, 0) > 3:
                reducer = umap.UMAP(
                    n_components=target_dim, 
                    n_neighbors=min(15, len(X)//4),
                    min_dist=0.1,
                    metric=self.system._physical_distance
                )
                method = "UMAP"
            elif X.shape[1] > 10:
                reducer = PCA(n_components=target_dim)
                method = "PCA"
            else:
                reducer = TSNE(
                    n_components=target_dim, 
                    perplexity=min(30, len(X)-1),
                    n_iter=1000,
                    metric=self.system._physical_distance
                )
                method = "t-SNE"
            
            reduced_points = reducer.fit_transform(X)
            
            # Создание нового измерения для редуцированного гиперкуба
            dim_name = f"ReducedSpace_{method}"
            reduced_range = (np.min(reduced_points), np.max(reduced_points))
            
            # Обновление системы с новым измерением
            self.system.dimensions[dim_name] = reduced_range
            self.system.dim_names.append(dim_name)
            
            # Переиндексация известных точек
            new_points = []
            for i, point in enumerate(self.system.known_points):
                new_point = point + reduced_points[i].tolist()
                new_points.append(new_point)
            
            self.system.known_points = new_points
            self.logger.info(f"Dimensionality reduced from {X.shape[1]}D to {target_dim}D using {method}")
            
            # Инициируем эволюцию топологии
            self.system.topology_engine.evolve_topology()
            
            return reduced_points
        
        except Exception as e:
            self.logger.error(f"Dimensionality reduction failed: {str(e)}")
            return None

    def boundary_topology_analysis(self):
        """
        Расширенный анализ граничных данных с использованием квантово-топологических методов
        Возвращает словарь с ключевыми характеристиками
        """
        analysis = {}
        
        # 1. Анализ связности с квантовой верностью
        connectivity = self._calculate_quantum_connectivity()
        analysis['quantum_connectivity'] = connectivity
        
        # 2. Топологические дефекты с ансамблевым анализом
        defects = self._detect_ensemble_defects()
        analysis['ensemble_defects'] = defects
        
        # 3. Энтропийные характеристики с учетом квантовой когерентности
        entropy = self._calculate_quantum_entropy()
        analysis['quantum_entropy'] = entropy
        
        # 4. Анализ стабильности ансамбля систем
        if hasattr(self.system, 'topological_ensemble_interface'):
            stability = self._analyze_ensemble_stability()
            analysis['ensemble_stability'] = stability
        
        self.logger.info("Advanced boundary topology analysis completed")
        return analysis

    def _calculate_quantum_connectivity(self):
        """Вычисление мер связности с использованием квантовой верности"""
        if not self.system.critical_points:
            return {}
            
        points = np.array([cp['point'] for cp in self.system.critical_points])
        
        # Создание квантовых состояний для критических точек
        quantum_states = []
        for point in points:
            state_vector = self.system.topology_engine.quantum_core.get_topology_vector(point)
            circuit = self.system.topology_engine.quantum_core.create_entanglement_circuit(state_vector)
            quantum_states.append(circuit)
        
        # Расчет матрицы квантовой верности
        fidelity_matrix = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                fidelity = self.system.topology_engine.quantum_core.calculate_topological_fidelity(
                    quantum_states[i], quantum_states[j]
                )
                fidelity_matrix[i, j] = fidelity
                fidelity_matrix[j, i] = fidelity
        
        # Анализ квантовой связности
        connectivity = {
            'average_fidelity': np.mean(fidelity_matrix),
            'max_fidelity': np.max(fidelity_matrix),
            'min_fidelity': np.min(fidelity_matrix),
            'entanglement_entropy': self._calculate_entanglement_entropy(fidelity_matrix)
        }
        return connectivity

    def _calculate_entanglement_entropy(self, fidelity_matrix):
        """Расчет энтропии запутанности на основе матрицы верности"""
        eigenvalues = np.linalg.eigvalsh(fidelity_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return entropy

    def _detect_ensemble_defects(self):
        """Обнаружение топологических дефектов с учетом ансамбля систем"""
        defects = {'monopoles': [], 'strings': [], 'domain_walls': []}
        
        if not self.system.critical_points:
            return defects
            
        # Кластеризация с учетом ансамбля
        points = np.array([cp['point'] for cp in self.system.critical_points])
        values = np.array([cp['value'] for cp in self.system.critical_points])
        
        clustering = DBSCAN(eps=0.1, min_samples=3).fit(points)
        labels = clustering.labels_
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue
                
            cluster_points = points[labels == label]
            cluster_values = values[labels == label]
            
            # Классификация дефектов
            size = len(cluster_points)
            dim = cluster_points.shape[1]
            
            if size == 1:
                defect_type = 'monopoles'
            elif dim == 1 or size < 5:
                defect_type = 'strings'
            else:
                defect_type = 'domain_walls'
            
            # Анализ стабильности в ансамбле систем
            if hasattr(self.system, 'topological_ensemble_interface'):
                stability = []
                for system_id in self.system.topological_ensemble_interface.parallel_systems:
                    system = self.system.topological_ensemble_interface.parallel_systems[system_id]
                    for point in cluster_points:
                        stability.append(self._measure_point_stability(point, system))
                avg_stability = np.mean(stability)
            else:
                avg_stability = 1.0
                
            defects[defect_type].append({
                'points': cluster_points.tolist(),
                'stability': avg_stability
            })
        
        return defects

    def _measure_point_stability(self, point, system):
        """Измерение стабильности точки в параллельной системе"""
        try:
            # Получаем значение в основной системе
            main_value = self.system.physical_query_dict(
                {dim: point[i] for i, dim in enumerate(self.system.dim_names)}
            )
            
            # Получаем значение в параллельной системе
            parallel_value = system.physical_query_dict(
                {dim: point[i] for i, dim in enumerate(system.dim_names)}
            )
            
            # Вычисляем относительную стабильность
            if np.isnan(main_value) or np.isnan(parallel_value):
                return 0
            return 1 - abs(main_value - parallel_value) / (abs(main_value) + 1e-10)
        except:
            return 0

    def _calculate_quantum_entropy(self):
        """Расчет энтропийных характеристик с учетом квантовой когерентности"""
        entropy_metrics = self._calculate_entropy_metrics()
        
        # Добавляем квантовую энтропию
        if self.system.quantum_model:
            entropy_metrics['quantum_entropy'] = self._calculate_quantum_coherence_entropy()
        
        # Энтропия ансамбля систем
        if hasattr(self.system, 'topological_ensemble_interface'):
            entropy_metrics['ensemble_entropy'] = self._calculate_ensemble_entropy()
        
        return entropy_metrics

    def _calculate_quantum_coherence_entropy(self):
        """Расчет энтропии квантовой когерентности"""
        try:
            # Создание эталонного запутанного состояния
            ref_circuit = QuantumCircuit(2)
            ref_circuit.h(0)
            ref_circuit.cx(0, 1)
            
            # Измерение состояния квантовой модели
            quantum_state = self.system.quantum_model.quantum_instance.execute(ref_circuit).result().get_statevector()
            
            # Расчет матрицы плотности
            density_matrix = np.outer(quantum_state, np.conj(quantum_state))
            
            # Расчет энтропии фон Неймана
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            eigenvalues = eigenvalues[eigenvalues > 0]
            entropy = -np.sum(eigenvalues * np.log(eigenvalues))
            return entropy
        except:
            return 0

    def _calculate_ensemble_entropy(self):
        """Расчет энтропии ансамбля систем"""
        system_ids = list(self.system.topological_ensemble_interface.parallel_systems.keys())
        if len(system_ids) < 2:
            return 0
            
        # Сравнение систем попарно
        divergences = []
        for i in range(len(system_ids)):
            for j in range(i+1, len(system_ids)):
                comparison = self.system.topological_ensemble_interface.compare_systems(
                    system_ids[i], system_ids[j]
                )
                divergence = 1 - comparison['stability_ratio']
                divergences.append(divergence)
        
        return np.mean(divergences) if divergences else 0

    def topological_quantum_optimization(self, backend='simulator', depth=None):
        """
        Квантовая оптимизация через запутывание состояний
        с адаптацией под текущую топологию системы
        """
        if depth is None:
            depth = self.quantum_depth
            
        try:
            if len(self.system.known_points) < 5:
                self.logger.warning("Insufficient points for quantum optimization")
                return False
                
            X = np.array(self.system.known_points)
            y = np.array(self.system.known_values)
            
            # Динамическая нормализация с учетом топологии
            X = self._topology_aware_normalization(X)
            y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)
            
            # Создание квантовой схемы с адаптивной глубиной
            num_qubits = min(8, X.shape[1])
            feature_map = EfficientSU2(num_qubits, reps=2)
            ansatz = EfficientSU2(num_qubits, reps=depth)
            quantum_circuit = feature_map.compose(ansatz)
            
            # Создание квантовой нейронной сети
            qnn = CircuitQNN(
                circuit=quantum_circuit,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
                input_gradients=True,
                quantum_instance=Aer.get_backend(backend)
            )
            
            # Обучение с топологической регуляризацией
            vqc = VQC(
                num_qubits=num_qubits,
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=SPSA(maxiter=50),
                quantum_instance=Aer.get_backend(backend),
                callback=self._quantum_optimization_callback
            )
            
            vqc.fit(X, y)
            
            # Перенос квантовых знаний в систему
            self._transfer_quantum_knowledge(vqc, X, y)
            
            # Инициируем эволюцию топологии
            self.system.topology_engine.evolve_topology()
            
            self.logger.info(f"Quantum entanglement optimization completed with depth={depth}")
            return True
        
        except Exception as e:
            self.logger.error(f"Quantum optimization failed: {str(e)}")
            return False

    def _topology_aware_normalization(self, X):
        """Нормализация данных с учетом топологических особенностей"""
        # Используем критические точки как опорные
        if self.system.critical_points:
            critical_points = np.array([cp['point'] for cp in self.system.critical_points])
            mins = np.min(np.vstack([np.min(X, axis=0), np.min(critical_points, axis=0)]), axis=0)
            maxs = np.max(np.vstack([np.max(X, axis=0), np.max(critical_points, axis=0)]), axis=0)
        else:
            mins = np.min(X, axis=0)
            maxs = np.max(X, axis=0)
            
        return (X - mins) / (maxs - mins + 1e-10)

    def _quantum_optimization_callback(self, weights, loss):
        """Коллбэк для отслеживания процесса квантовой оптимизации"""
        self.logger.debug(f"Quantum optimization loss: {loss:.4f}")
        
        # Адаптивная корректировка на основе потерь
        if loss > 0.5 and not hasattr(self, 'adjustment_made'):
            self.logger.info("High loss detected, adjusting topology...")
            self.system.topology_engine.evolve_topology()
            self.adjustment_made = True

    def _transfer_quantum_knowledge(self, vqc, X, y):
        """Перенос квантовых знаний в классическую модель"""
        quantum_predictions = vqc.predict(X)
        
        for i, point in enumerate(self.system.known_points):
            # Комбинируем квантовые и классические знания
            quantum_value = quantum_predictions[i]
            classical_value = self.system.known_values[i]
            
            # Вес основан на квантовой неопределенности
            quantum_uncertainty = abs(quantum_value - classical_value)
            weight = np.exp(-quantum_uncertainty)
            
            new_value = weight * quantum_value + (1 - weight) * classical_value
            self.system.known_values[i] = new_value
            
        # Перестраиваем GP модель с новыми значениями
        self.system._build_gaussian_process()

    def collective_behavior_detection(self, threshold=0.15):
        """
        Обнаружение коллективных свойств с учетом ансамблевых взаимодействий
        """
        emergent_properties = []
        
        # 1. Нелинейные взаимодействия с топологическим анализом
        interaction_strength = self._measure_nonlinear_interactions()
        if interaction_strength > threshold:
            emergent_properties.append({
                'type': 'topological_nonlinearity',
                'strength': interaction_strength,
                'description': 'Сильные нелинейные взаимодействия с топологической структурой'
            })
        
        # 2. Критические точки с ансамблевой стабильностью
        if self.system.critical_points and len(self.system.critical_points) > 3:
            clusters = self._cluster_critical_points()
            if len(set(clusters)) > 1:
                stability = self._measure_critical_point_stability(clusters)
                emergent_properties.append({
                    'type': 'ensemble_phase_clusters',
                    'count': len(set(clusters)),
                    'stability': stability,
                    'description': 'Кластеры фазовых переходов с ансамблевой стабильностью'
                })
        
        # 3. Квантовая когерентность в ансамбле
        quantum_coherence = self._measure_quantum_coherence()
        if quantum_coherence > threshold:
            ensemble_coherence = self._measure_ensemble_coherence()
            emergent_properties.append({
                'type': 'ensemble_quantum_coherence',
                'coherence': quantum_coherence,
                'ensemble_coherence': ensemble_coherence,
                'description': 'Квантовая когерентность, сохраняющаяся в ансамбле систем'
            })
        
        # 4. Топологическая эмерджентность
        entropy_metrics = self._calculate_entropy_metrics()
        entropy_ratio = entropy_metrics.get('entropy_ratio', 0)
        if entropy_ratio > 0.7:
            emergent_properties.append({
                'type': 'topological_emergence',
                'entropy_ratio': entropy_ratio,
                'description': 'Эмерджентные свойства, вызванные сложной топологией'
            })
        
        self.logger.info(f"Detected {len(emergent_properties)} emergent properties")
        return emergent_properties

    def _measure_critical_point_stability(self, clusters):
        """Измерение стабильности критических точек в ансамбле"""
        if not hasattr(self.system, 'topological_ensemble_interface'):
            return 1.0
            
        stability_scores = []
        for cluster_id in set(clusters):
            if cluster_id == -1:
                continue
                
            cluster_points = [self.system.critical_points[i]['point'] 
                             for i, cid in enumerate(clusters) if cid == cluster_id]
            
            cluster_stability = []
            for point in cluster_points:
                for system_id in self.system.topological_ensemble_interface.parallel_systems:
                    system = self.system.topological_ensemble_interface.parallel_systems[system_id]
                    cluster_stability.append(self._measure_point_stability(point, system))
            
            stability_scores.append(np.mean(cluster_stability))
        
        return np.mean(stability_scores) if stability_scores else 1.0

    def _measure_ensemble_coherence(self):
        """Измерение квантовой когерентности в ансамбле"""
        if not hasattr(self.system, 'topological_ensemble_interface'):
            return 0
            
        coherence_scores = []
        for system_id in self.system.topological_ensemble_interface.parallel_systems:
            system = self.system.topological_ensemble_interface.parallel_systems[system_id]
            coherence = self._measure_quantum_coherence_for_system(system)
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0

    def _measure_quantum_coherence_for_system(self, system):
        """Измерение квантовой когерентности для конкретной системы"""
        if not system.quantum_model:
            return 0
        
        try:
            ref_circuit = QuantumCircuit(2)
            ref_circuit.h(0)
            ref_circuit.cx(0, 1)
            
            quantum_state = system.quantum_model.quantum_instance.execute(ref_circuit).result().get_statevector()
            fidelity = state_fidelity(quantum_state, ref_circuit)
            return fidelity
        except:
            return 0

    def fundamental_constraint_integration(self, constraint_type='causal'):
        """
        Интеграция фундаментальных ограничений с динамической адаптацией
        """
        # Применяем ограничение
        if constraint_type == 'causal':
            self.system.physical_constraint = self._causal_constraint
        elif constraint_type == 'deterministic':
            self.system.physical_constraint = self._deterministic_constraint
        elif constraint_type == 'topological':
            self.system.physical_constraint = self._topological_constraint
        elif constraint_type == 'ensemble_consistent':
            self.system.physical_constraint = self._ensemble_consistency_constraint
        
        self.logger.info(f"Applied {constraint_type} fundamental constraint")
        
        # Эволюционируем топологию под новое ограничение
        self.system.topology_engine.evolve_topology()

    def _ensemble_consistency_constraint(self, params):
        """
        Ограничение согласованности ансамбля: значения должны быть
        согласованы между параллельными системами
        """
        if not hasattr(self.system, 'topological_ensemble_interface'):
            return True
            
        main_value = self.system.physical_query_dict(params)
        if np.isnan(main_value):
            return False
            
        for system_id in self.system.topological_ensemble_interface.parallel_systems:
            system = self.system.topological_ensemble_interface.parallel_systems[system_id]
            sys_value = system.physical_query_dict(params)
            
            if np.isnan(sys_value) or abs(main_value - sys_value) > 0.1 * abs(main_value):
                return False
                
        return True

    def ensemble_guided_optimization(self, target_properties, num_systems=5):
        """
        Оптимизация через создание и анализ ансамбля систем
        """
        if not hasattr(self.system, 'topological_ensemble_interface'):
            self.logger.warning("Topological ensemble interface not available")
            return None
            
        # Создаем параллельные системы
        for i in range(num_systems):
            modification_rules = self._generate_modification_rules()
            system_id = f"optimization_{i}"
            self.system.topological_ensemble_interface.create_parallel_system(
                system_id, modification_rules)
        
        # Оцениваем системы на соответствие целевым свойствам
        best_system = None
        best_score = -np.inf
        
        for system_id, system in self.system.topological_ensemble_interface.parallel_systems.items():
            if system_id.startswith("optimization_"):
                score = self._evaluate_system(system, target_properties)
                if score > best_score:
                    best_score = score
                    best_system = system
        
        # Переносим лучшие точки в основную систему
        if best_system:
            self._transfer_knowledge(best_system)
            return best_system
        return None

    def _generate_modification_rules(self):
        """Генерация правил модификации для параллельной системы"""
        rules = {}
        for dim in self.system.dim_names:
            if np.random.rand() > 0.7:  # 30% chance to modify a dimension
                rule_type = np.random.choice(['shift', 'scale', 'invert'])
                
                if rule_type == 'shift':
                    amount = np.random.uniform(-0.5, 0.5)
                    rules[dim] = {'type': 'shift', 'amount': amount}
                elif rule_type == 'scale':
                    factor = np.random.uniform(0.5, 2.0)
                    rules[dim] = {'type': 'scale', 'factor': factor}
                else:  # invert
                    rules[dim] = {'type': 'invert'}
        return rules

    def _evaluate_system(self, system, target_properties):
        """Оценка системы на соответствие целевым свойствам"""
        score = 0
        
        # Соответствие топологии
        if 'betti_numbers' in target_properties:
            system.calculate_topological_invariants()
            betti_match = 0
            for dim, target_val in target_properties['betti_numbers'].items():
                system_val = system.topological_invariants['betti_numbers'].get(int(dim), 0)
                betti_match += 1 - abs(target_val - system_val) / (target_val + 1)
            score += betti_match
        
        # Соответствие симметриям
        if 'symmetry_level' in target_properties:
            system.find_symmetries()
            symmetry_match = 1 - abs(len(system.symmetries) - target_properties['symmetry_level']) / 10
            score += symmetry_match
        
        # Соответствие квантовой когерентности
        if 'quantum_coherence' in target_properties:
            coherence = self._measure_quantum_coherence_for_system(system)
            coherence_match = 1 - abs(coherence - target_properties['quantum_coherence'])
            score += coherence_match
        
        return score

    def _transfer_knowledge(self, source_system):
        """Перенос знаний из параллельной системы в основную"""
        # Перенос критических точек
        for cp in source_system.critical_points:
            self.system.critical_points.append(cp)
        
        # Перенос точек данных
        for point, value in zip(source_system.known_points, source_system.known_values):
            params = {dim: point[i] for i, dim in enumerate(self.system.dim_names)}
            self.system.add_known_point(params, value)
        
        # Перенос симметрий
        for dim, sym_data in source_system.symmetries.items():
            if dim not in self.system.symmetries:
                self.system.symmetries[dim] = sym_data
        
        self.logger.info(f"Transferred knowledge from parallel system")

    def topology_guided_optimization(self, target_betti):
        """
        Оптимизация с направленной эволюцией топологии
        """
        # Инженерное изменение топологии
        self.engineer_topology(target_betti)
        
        # Эволюционируем топологию
        self.system.topology_engine.evolve_topology()
        
        # Квантовая оптимизация на новой топологии
        self.topological_quantum_optimization()
        
        return self.system.topological_invariants['betti_numbers']

    # ========== Унаследованные методы с улучшениями ==========
    def _calculate_entropy_metrics(self):
        """Расчет энтропийных характеристик системы"""
        if not self.system.known_values:
            return {}
            
        values = np.array(self.system.known_values)
        
        # Информационная энтропия
        hist, bins = np.histogram(values, bins=20, density=True)
        hist = hist / np.sum(hist)
        shannon_entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # Топологическая энтропия
        if self.system.critical_points:
            critical_values = np.array([cp['value'] for cp in self.system.critical_points])
            crit_hist = np.histogram(critical_values, bins=5)[0]
            crit_hist = crit_hist / np.sum(crit_hist)
            topological_entropy = -np.sum(crit_hist * np.log(crit_hist + 1e-10))
        else:
            topological_entropy = 0
        
        return {
            'shannon_entropy': shannon_entropy,
            'topological_entropy': topological_entropy,
            'entropy_ratio': topological_entropy / (shannon_entropy + 1e-10)
        }

    def _measure_nonlinear_interactions(self):
        """Измерение силы нелинейных взаимодействий между измерениями"""
        interaction_strength = 0
        num_interactions = 0
        
        for i, dim1 in enumerate(self.system.dim_names):
            for j, dim2 in enumerate(self.system.dim_names):
                if i >= j or not self.system.known_points:
                    continue
                
                # Выборка точек с вариацией по двум измерениям
                points = []
                values = []
                for k, point in enumerate(self.system.known_points):
                    if (abs(point[i] - np.mean([p[i] for p in self.system.known_points]))) > 0.1 and \
                       (abs(point[j] - np.mean([p[j] for p in self.system.known_points]))) > 0.1:
                        points.append([point[i], point[j]])
                        values.append(self.system.known_values[k])
                
                if len(points) < 10:
                    continue
                
                # Подгонка поверхности
                X = np.array(points)
                y = np.array(values)
                
                # Сравнение линейной и нелинейной моделей
                linear_error = self._fit_model(X, y, degree=1)
                nonlinear_error = self._fit_model(X, y, degree=2)
                
                # Относительное улучшение
                improvement = (linear_error - nonlinear_error) / linear_error
                interaction_strength += max(0, improvement)
                num_interactions += 1
        
        return interaction_strength / num_interactions if num_interactions > 0 else 0

    def _fit_model(self, X, y, degree=1):
        """Аппроксимация поверхности полиномом заданной степени"""
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        return mean_squared_error(y, y_pred)

    def _cluster_critical_points(self):
        """Кластеризация критических точек"""
        if not self.system.critical_points:
            return []
            
        points = np.array([cp['point'] for cp in self.system.critical_points])
        clustering = DBSCAN(eps=0.2, min_samples=2).fit(points)
        return clustering.labels_

    def _measure_quantum_coherence(self):
        """Измерение квантовой когерентности в системе"""
        if not self.system.quantum_model:
            return 0
        
        try:
            # Создание эталонных состояний
            ref_circuit = QuantumCircuit(2)
            ref_circuit.h(0)
            ref_circuit.cx(0, 1)
            
            # Измерение состояния квантовой модели
            quantum_state = self.system.quantum_model.quantum_instance.execute(ref_circuit).result().get_statevector()
            
            # Расчет фиделити
            fidelity = state_fidelity(quantum_state, ref_circuit)
            return fidelity
        
        except:
            return 0

    def _causal_constraint(self, params):
        """Проверка причинно-следственных отношений"""
        time_dims = [dim for dim in self.system.dim_names if 'time' in dim.lower()]
        for dim in time_dims:
            if dim in params:
                prev_values = [p[self.system.dim_names.index(dim)] 
                              for p in self.system.known_points if dim in p]
                if prev_values and params[dim] < max(prev_values):
                    return False
        return True

    def _deterministic_constraint(self, params):
        """Проверка детерминированности"""
        similar_points = []
        for i, point in enumerate(self.system.known_points):
            distance = self.system._physical_distance(
                [point[self.system.dim_names.index(dim)] for dim in params.keys()],
                [params[dim] for dim in params.keys()]
            )
            if distance < 0.01:
                similar_points.append(self.system.known_values[i])
        
        if similar_points:
            avg_value = np.mean(similar_points)
            std_value = np.std(similar_points)
            if std_value > 0.1 * avg_value:
                return False
        return True

    def _topological_constraint(self, params):
        """Проверка топологического соответствия"""
        if not self.system.topological_compression:
            return True
        
        estimated_value = self.system._estimate_from_topology(
            [params[dim] for dim in self.system.dim_names]
        )
        actual_value = self.system.physical_query_dict(params)
        
        return abs(estimated_value - actual_value) < 0.1 * abs(actual_value)

# ===================================================================
# Основной класс PhysicsHypercubeSystem
# ===================================================================
class PhysicsHypercubeSystem:
    def __init__(self, dimensions, resolution=100, extrapolation_limit=0.2, 
                 physical_constraint=None, collision_tolerance=0.05, 
                 uncertainty_slope=0.1, parent_hypercube=None):
        """
        dimensions: словарь измерений и их диапазонов
        resolution: точек на измерение (для визуализации)
        extrapolation_limit: максимальное относительное отклонение для экстраполяции
        physical_constraint: функция проверки физической реализуемости точки
        collision_tolerance: допуск для коллизионных линий
        uncertainty_slope: коэффициент для оценки неопределенности
        parent_hypercube: родительский гиперкуб для иерархии
        """
        self.dimensions = dimensions
        self.dim_names = list(dimensions.keys())
        self.resolution = resolution
        self.extrapolation_limit = extrapolation_limit
        self.physical_constraint = physical_constraint
        self.collision_tolerance = collision_tolerance
        self.uncertainty_slope = uncertainty_slope
        self.phase_transition_func = None
        
        # Иерархия гиперкубов
        self.parent_hypercube = parent_hypercube
        self.child_hypercubes = []
        
        # Топологические инварианты
        self.topological_invariants = {}
        self.symmetries = {}
        self.critical_points = []
        
        # Квантовые параметры
        self.quantum_optimization_enabled = False
        self.quantum_backend = None
        self.quantum_model = None
        
        # Топологическое представление
        self.topological_compression = False
        self.boundary_data = {}
        
        # Расширенная система измерений
        self.dimension_types = {}
        for dim in dimensions:
            if isinstance(dimensions[dim], tuple):
                self.dimension_types[dim] = 'continuous'
            elif isinstance(dimensions[dim], list):
                self.dimension_types[dim] = 'categorical'
            else:
                raise ValueError(f"Invalid dimension specification for {dim}")
        
        # Хранилища данных
        self.known_points = []
        self.known_values = []
        self.collision_lines = []
        self.gp_model = None
        self.gp_likelihood = None
        
        # Ресурсные менеджеры
        self.gpu_manager = GPUComputeManager()
        self.smart_cache = SmartCache()
        
        # Настройка системы
        self._setup_logging()
        self._auto_configure()
        
        self.logger.info("PhysicsHypercubeSystem initialized with full GPU and cache support")
    
    # ---------------------- ИНТЕГРИРОВАННЫЕ МЕТОДЫ ----------------------
    
    def _auto_configure(self):
        """Автоматическая настройка параметров системы под оборудование"""
        try:
            total_mem = psutil.virtual_memory().total
            if total_mem < 8e9:  # < 8GB RAM
                self.resolution = 50
                self.smart_cache.max_size = 1000
            elif total_mem < 16e9:  # < 16GB RAM
                self.resolution = 100
                self.smart_cache.max_size = 5000
            else:  # >= 16GB RAM
                self.resolution = 200
                self.smart_cache.max_size = 20000
                
            self.logger.info(f"Auto-configured: resolution={self.resolution}, cache_limit={self.smart_cache.max_size}")
        except Exception as e:
            self.logger.error(f"Auto-configuration failed: {str(e)}")
    
    def _setup_logging(self):
        """Настройка системы журналирования"""
        self.logger = logging.getLogger("PhysicsHypercubeSystem")
        self.logger.setLevel(logging.INFO)
        
        # Ротация логов (10 файлов по 1MB)
        file_handler = RotatingFileHandler(
            "physics_hypercube.log", maxBytes=1e6, backupCount=10
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Консольный вывод
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def add_known_point(self, point, value):
        """Добавление известной точки в гиперкуб"""
        ordered_point = [point[dim] for dim in self.dim_names]
        self.known_points.append(ordered_point)
        self.known_values.append(value)
        
        # Кэшируем как постоянное значение
        params_tuple = tuple(ordered_point)
        self.smart_cache.set(params_tuple, value, is_permanent=True)
        
        self.logger.info(f"Added known point: {point} = {value}")
        self._build_gaussian_process()
    
    def _build_gaussian_process(self):
        """Построение модели гауссовского процесса на GPU/CPU"""
        if len(self.known_points) < 3:
            return
            
        X = np.array(self.known_points)
        y = np.array(self.known_values)
        
        # Использование квантовой оптимизации при включенном флаге
        if self.quantum_optimization_enabled:
            def quantum_train_task(device):
                try:
                    from qiskit_machine_learning.kernels import QuantumKernel
                    from qiskit.circuit.library import ZZFeatureMap
                    
                    # Создание квантового ядра
                    feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
                    quantum_kernel = QuantumKernel(
                        feature_map=feature_map, 
                        quantum_instance=self.quantum_backend
                    )
                    
                    # Квантовая GP регрессия
                    from qiskit_machine_learning.algorithms import QGPR
                    qgpr = QGPR(quantum_kernel=quantum_kernel)
                    qgpr.fit(X, y)
                    
                    # Сохранение модели
                    self.quantum_model = qgpr
                    return True
                except ImportError:
                    self.logger.warning("Quantum libraries not available. Falling back to classical GP.")
                    self.quantum_optimization_enabled = False
                    return self._build_gaussian_process()
            
            self.gpu_manager.execute(quantum_train_task)
            self.logger.info("Quantum Gaussian Process model built")
        else:
            # Классическая реализация
            def train_task(device):
                train_x = torch.tensor(X, dtype=torch.float32).to(device)
                train_y = torch.tensor(y, dtype=torch.float32).to(device)
                
                # Инициализация модели и likelihood
                self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
                self.gp_model = ExactGPModel(
                    train_x, train_y, self.gp_likelihood, kernel="RBF"
                ).to(device)
                
                # Обучение модели
                self.gp_model.train()
                self.gp_likelihood.train()
                
                optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)
                
                training_iter = 100
                for i in range(training_iter):
                    optimizer.zero_grad()
                    output = self.gp_model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.step()
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"GP Iteration {i+1}/{training_iter} - Loss: {loss.item():.4f}")
                
                return True
            
            # Запуск обучения с управлением ресурсами
            self.gpu_manager.execute(train_task)
            self.logger.info("Classical Gaussian Process model rebuilt")

    def _gp_predict(self, point, return_std=False):
        """Предсказание с использованием GP модели"""
        # Если включена квантовая оптимизация и модель доступна
        if self.quantum_optimization_enabled and hasattr(self, 'quantum_model') and self.quantum_model is not None:
            point = np.array([point])
            if return_std:
                mean, std = self.quantum_model.predict(point, return_std=True)
                return mean[0], std[0]
            else:
                return self.quantum_model.predict(point)[0]
        
        # Классическая реализация
        if self.gp_model is None or self.gp_likelihood is None:
            return (np.nan, np.nan) if return_std else np.nan
            
        def predict_task(device):
            self.gp_model.eval()
            self.gp_likelihood.eval()
            
            test_x = torch.tensor([point], dtype=torch.float32).to(device)
            
            # Правильный вызов модели
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.gp_likelihood(self.gp_model(test_x))
                mean = observed_pred.mean.item()
                std = observed_pred.stddev.item()
                return mean, std
        
        mean, std = self.gpu_manager.execute(predict_task)
        
        # Применение физических ограничений
        if self.physical_constraint is not None:
            params = {dim: point[i] for i, dim in enumerate(self.dim_names)}
            if not self.physical_constraint(params):
                mean = np.nan
                std = np.nan
        
        # Применение ограничения на положительность энергии
        if mean < 0:
            self.logger.debug(f"Negative energy detected at {point}, clipping to 0")
            mean = 0.0
            std = max(std, 0.1)  # Увеличиваем неопределенность
        
        return (mean, std) if return_std else mean
    
    def _physical_distance(self, point1, point2):
        """Вычисление физически осмысленного расстояния с учетом безразмерных отношений"""
        # Преобразование в безразмерные величины
        ratios = []
        for i, dim in enumerate(self.dim_names):
            dim_min, dim_max = self.dimensions[dim]
            range_val = dim_max - dim_min
            val1 = (point1[i] - dim_min) / range_val
            val2 = (point2[i] - dim_min) / range_val
            ratios.append(abs(val1 - val2))
        
        # Учет иерархии взаимодействий
        weights = {
            'gravitational': 0.8,
            'electromagnetic': 0.6,
            'strong': 0.4,
            'weak': 0.2
        }
        
        # Вычисление взвешенного расстояния
        weighted_sum = 0.0
        total_weight = 0.0
        for i, dim in enumerate(self.dim_names):
            weight = weights.get(dim, 0.5)  # Значение по умолчанию
            weighted_sum += weight * ratios[i] ** 2
            total_weight += weight
        
        return np.sqrt(weighted_sum / total_weight) if total_weight > 0 else np.linalg.norm(np.array(point1) - np.array(point2))
    
    def physical_query(self, params_tuple, return_std=False):
        """Запрос значения физического закона с интеллектуальным кэшированием и оценкой неопределенности"""
        # Проверка кэша для значения
        cached_value = self.smart_cache.get(params_tuple)
        if cached_value is not None and not return_std:
            return cached_value
        
        # Проверка кэша для неопределенности
        cache_key_std = params_tuple + ('_std',)
        cached_std = self.smart_cache.get(cache_key_std)
        if return_std and cached_value is not None and cached_std is not None:
            return cached_value, cached_std
        
        # Преобразуем кортеж обратно в словарь
        params = {dim: params_tuple[i] for i, dim in enumerate(self.dim_names)}
        
        # Проверка физических ограничений
        if self.physical_constraint is not None and not self.physical_constraint(params):
            result = np.nan
            std_dev = np.nan
        else:
            point = [params[dim] for dim in self.dim_names]
            
            # 1. Проверка коллизионных линий
            result = None
            std_dev = None
            
            # Поиск релевантных линий в окрестности
            relevant_lines = self.find_relevant_lines(point)
            for line, distance in relevant_lines:
                t, line_distance = self._project_to_line(point, line)
                
                # Если точка близка к линии, используем интерполяцию
                if line_distance < self.collision_tolerance:
                    if line['values'] is not None:
                        # Интерполяция значений вдоль линии
                        t_values = np.linspace(0, 1, len(line['values']))
                        result = np.interp(t, t_values, line['values'])
                        # Оценка неопределенности через расстояние до линии
                        std_dev = line_distance * self.uncertainty_slope
                        break
            
            # 2. Использование GP модели
            if result is None and self.gp_model is not None and self._is_within_extrapolation_limit(point):
                if return_std:
                    result, std_dev = self._gp_predict(point, return_std=True)
                else:
                    result = self._gp_predict(point)
            
            # 3. Поиск ближайшей известной точки
            if result is None and self.known_points:
                X = np.array(self.known_points)
                distances = [self._physical_distance(point, p) for p in self.known_points]
                min_distance = min(distances)
                idx = np.argmin(distances)
                result = self.known_values[idx]
                # Оценка неопределенности через расстояние
                std_dev = min_distance * self.uncertainty_slope
        
        # Применение фазовых переходов
        if self.phase_transition_func is not None:
            phase_result = self.phase_transition_func(params)
            if phase_result is not None:
                result = phase_result
                if std_dev is None:
                    std_dev = 0.1  # Стандартная неопределенность для фазового перехода
        
        # Сохранение в кэш
        self.smart_cache.set(params_tuple, result)
        if return_std and std_dev is not None:
            self.smart_cache.set(cache_key_std, std_dev)
            return result, std_dev
        
        return result
    
    # ---------------------- НОВЫЕ И УЛУЧШЕННЫЕ МЕТОДЫ ----------------------
    
    def physical_query_dict(self, params, return_std=False):
        """Запрос значения физического закона (через словарь) с возможностью возврата неопределенности"""
        # Проверка типов измерений
        for dim, value in params.items():
            if self.dimension_types[dim] == 'categorical':
                if value not in self.dimensions[dim]:
                    raise ValueError(f"Invalid category {value} for dimension {dim}")
        
        # Преобразование категориальных значений в числовые индексы
        query_params = []
        for dim in self.dim_names:
            value = params[dim]
            if self.dimension_types[dim] == 'categorical':
                # Кодирование категории как индекса
                query_params.append(self.dimensions[dim].index(value))
            else:
                query_params.append(value)
        
        # Создаем хешируемый кортеж значений
        params_tuple = tuple(query_params)
        self.logger.debug(f"Query: {params}")
        
        if return_std:
            result, std = self.physical_query(params_tuple, return_std=True)
            self.logger.debug(f"Result: {result:.4f} ± {std:.4f}")
            return result, std
        else:
            result = self.physical_query(params_tuple)
            self.logger.debug(f"Result: {result}")
            return result
    
    def add_collision_line(self, base_point, direction_vector, values=None):
        """
        Добавление коллизионной линии
        values: опциональные значения вдоль линии
        """
        line = {
            'base': [base_point[dim] for dim in self.dim_names],
            'direction': [direction_vector[dim] for dim in self.dim_names],
            'values': values
        }
        self.collision_lines.append(line)
        self.logger.info(f"Added collision line: base={base_point}, direction={direction_vector}")
        
        # Если значения не предоставлены, аппроксимируем их
        if values is None and self.gp_model is not None:
            self._approximate_line_values(line)
    
    def find_relevant_lines(self, target_point, radius=0.1):
        """Поиск коллизионных линий в физической окрестности точки"""
        relevant_lines = []
        for line in self.collision_lines:
            # Вычисление минимального расстояния до линии
            t, distance = self._project_to_line(target_point, line)
            
            # Физическая метрика расстояния
            base_point = np.array(line['base'])
            direction = np.array(line['direction'])
            closest_point = base_point + t * direction
            phys_dist = self._physical_distance(target_point, closest_point)
            
            if phys_dist < radius:
                relevant_lines.append((line, phys_dist))
        
        # Сортировка по расстоянию
        relevant_lines.sort(key=lambda x: x[1])
        return relevant_lines
    
    def set_phase_transition(self, phase_func):
        """Установка функции для обработки фазовых переходов"""
        self.phase_transition_func = phase_func
        self.logger.info("Phase transition function set")
    
    def validate_physical_law(self, test_points, expected_func, tolerance=0.05):
        """
        Валидация системы на известном физическом законе
        test_points: список точек для тестирования
        expected_func: функция, возвращающая ожидаемое значение
        tolerance: допустимая относительная погрешность
        """
        errors = []
        for point in test_points:
            # Вычисляем предсказанное значение
            predicted = self.physical_query_dict(point)
            
            # Вычисляем ожидаемое значение
            expected = expected_func(point)
            
            # Вычисляем относительную ошибку
            if np.isnan(predicted) or np.isnan(expected):
                error = np.nan
            elif abs(expected) > 1e-6:
                error = abs(predicted - expected) / abs(expected)
            else:
                error = abs(predicted - expected)
                
            errors.append(error)
            
            self.logger.debug(f"Validation at {point}: Predicted={predicted:.4f}, Expected={expected:.4f}, Error={error:.2%}")
        
        # Статистика ошибок
        valid_errors = [e for e in errors if not np.isnan(e)]
        avg_error = np.mean(valid_errors) if valid_errors else np.nan
        max_error = np.max(valid_errors) if valid_errors else np.nan
        
        self.logger.info(f"Validation completed: Avg error={avg_error:.2%}, Max error={max_error:.2%}")
        return avg_error, max_error
    
    def find_functional_relations(self, max_terms=3):
        """Поиск аналитических зависимостей методом символьной регрессии"""
        try:
            from gplearn.genetic import SymbolicRegressor
        except ImportError:
            self.logger.warning("gplearn not installed, symbolic regression disabled")
            return None
        
        if len(self.known_points) < 10:
            self.logger.warning("Insufficient data for symbolic regression (min 10 points)")
            return None
        
        X = np.array(self.known_points)
        y = np.array(self.known_values)
        
        # Инициализация генетического алгоритма
        est = SymbolicRegressor(population_size=1000,
                                generations=20,
                                stopping_criteria=0.01,
                                p_crossover=0.7,
                                p_subtree_mutation=0.1,
                                p_hoist_mutation=0.05,
                                p_point_mutation=0.1,
                                max_samples=0.9,
                                verbose=1,
                                parsimony_coefficient=0.01,
                                random_state=0,
                                n_jobs=-1,
                                function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs'))
        
        est.fit(X, y)
        
        self.logger.info(f"Best functional relation: {est._program}")
        return est._program
    
    # ---------------------- НОВЫЕ МЕТОДЫ РАСШИРЕНИЙ ----------------------
    
    def topological_compression(self, compression_ratio=0.8):
        """
        Топологическое сжатие данных с сохранением граничной информации
        Сохраняет только граничные данные и топологические инварианты
        """
        self.topological_compression = True
        
        # Вычисление топологических инвариантов
        self.calculate_topological_invariants()
        
        # Поиск критических точек
        self.find_critical_points()
        
        # Сохранение критических точек
        self.boundary_data = {
            'topological_invariants': self.topological_invariants,
            'symmetries': self.symmetries,
            'critical_points': self.critical_points,
            'dimension_ranges': {dim: (min_val, max_val) 
                                for dim, (min_val, max_val) in self.dimensions.items()}
        }
        
        # Удаление внутренних данных (частичное сжатие)
        if compression_ratio < 1.0:
            keep_indices = np.random.choice(
                len(self.known_points),
                size=int(len(self.known_points) * compression_ratio),
                replace=False
            )
            self.known_points = [self.known_points[i] for i in keep_indices]
            self.known_values = [self.known_values[i] for i in keep_indices]
        
        self.logger.info(f"Hypercube compressed to boundary representation. Compression ratio: {compression_ratio}")
    
    def reconstruct_from_boundary(self, new_points=100):
        """
        Восстановление данных из граничного представления
        с использованием топологических инвариантов
        """
        if not self.topological_compression:
            self.logger.warning("Hypercube is not in compressed state")
            return
        
        # Восстановление данных на основе топологических инвариантов
        reconstructed_points = []
        reconstructed_values = []
        
        # Генерация новых точек на основе критических точек
        for _ in range(new_points):
            # Выбор случайной критической точки в качестве основы
            crit_point = random.choice(self.critical_points)
            point = []
            
            for i, dim in enumerate(self.dim_names):
                dim_min, dim_max = self.dimensions[dim]
                
                # Случайное смещение с учетом симметрий
                displacement = np.random.normal(0, 0.1 * (dim_max - dim_min))
                
                # Применение известных симметрий
                if dim in self.symmetries:
                    displacement *= self.symmetries[dim].get('factor', 1.0)
                
                new_val = crit_point['point'][i] + displacement
                new_val = np.clip(new_val, dim_min, dim_max)
                point.append(new_val)
            
            reconstructed_points.append(point)
            
            # Оценка значения на основе топологии
            value = self._estimate_from_topology(point)
            reconstructed_values.append(value)
        
        # Добавление восстановленных точек
        self.known_points.extend(reconstructed_points)
        self.known_values.extend(reconstructed_values)
        
        self.topological_compression = False
        self._build_gaussian_process()
        self.logger.info(f"Reconstructed {new_points} points from boundary data")
    
    def _estimate_from_topology(self, point):
        """Оценка значения на основе топологических инвариантов"""
        # Взвешенная сумма расстояний до критических точек
        total_weight = 0.0
        weighted_sum = 0.0
        
        for cp in self.critical_points:
            dist = self._physical_distance(point, cp['point'])
            weight = np.exp(-dist**2 / 0.1)
            weighted_sum += weight * cp['value']
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def calculate_topological_invariants(self):
        """Вычисление топологических инвариантов (чисел Бетти)"""
        try:
            from sklearn.preprocessing import MinMaxScaler
            from giotto_tda.homology import VietorisRipsPersistence
            from giotto_tda.diagrams import BettiCurve
        except ImportError:
            self.logger.warning("giotto-tda not installed, topological analysis disabled")
            return
        
        if len(self.known_points) < 10:
            self.logger.warning("Insufficient points for topological analysis")
            return
        
        # Нормализация данных
        scaler = MinMaxScaler()
        X = scaler.fit_transform(self.known_points)
        
        # Вычисление персистентных гомологий
        homology_dimensions = [0, 1, 2]  # Вычисляем до 2-мерных гомологий
        vr = VietorisRipsPersistence(
            homology_dimensions=homology_dimensions,
            n_jobs=-1
        )
        diagrams = vr.fit_transform([X])
        
        # Расчет кривых Бетти
        betti_curves = BettiCurve().fit(diagrams).transform(diagrams)
        
        # Сохранение инвариантов
        self.topological_invariants = {
            'betti_curves': betti_curves,
            'persistence_diagrams': diagrams
        }
        
        # Вычисление чисел Бетти
        self.topological_invariants['betti_numbers'] = {
            dim: int(np.sum(betti_curves[0][:, dim] > 0.1))
            for dim in homology_dimensions
        }
        
        self.logger.info(f"Topological invariants calculated: Betti numbers = {self.topological_invariants['betti_numbers']}")
    
    def find_symmetries(self, tolerance=0.05):
        """Поиск симметрий в данных гиперкуба"""
        symmetries = {}
        
        # Поиск непрерывных симметрий
        for dim in self.dim_names:
            if self.dimension_types[dim] != 'continuous':
                continue
                
            values = np.array([p[self.dim_names.index(dim)] for p in self.known_points])
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_outputs = np.array(self.known_values)[sorted_indices]
            
            # Проверка инвариантности относительно сдвига
            shift_invariant = True
            for i in range(1, len(sorted_values)):
                delta = sorted_values[i] - sorted_values[i-1]
                value_delta = abs(sorted_outputs[i] - sorted_outputs[i-1])
                if value_delta > tolerance * delta:
                    shift_invariant = False
                    break
            
            if shift_invariant:
                symmetries[dim] = {'type': 'shift', 'factor': 1.0}
        
        # Поиск дискретных симметрий (отражение)
        for dim in self.dim_names:
            if self.dimension_types[dim] != 'continuous':
                continue
                
            idx = self.dim_names.index(dim)
            dim_min, dim_max = self.dimensions[dim]
            center = (dim_min + dim_max) / 2
            
            symmetric_points = []
            for point in self.known_points:
                symmetric_point = point.copy()
                symmetric_point[idx] = 2*center - point[idx]
                symmetric_points.append(symmetric_point)
            
            # Проверка совпадения значений
            reflection_invariant = True
            for i, point in enumerate(self.known_points):
                symmetric_value = self.physical_query(symmetric_points[i])
                if abs(self.known_values[i] - symmetric_value) > tolerance:
                    reflection_invariant = False
                    break
            
            if reflection_invariant:
                symmetries[dim] = {'type': 'reflection', 'center': center}
        
        self.symmetries = symmetries
        self.logger.info(f"Found symmetries: {list(symmetries.keys())}")
        return symmetries
    
    def find_critical_points(self, threshold=0.2):
        """Поиск критических точек (фазовых переходов)"""
        self.critical_points = []
        
        if len(self.known_points) < 5:
            return
        
        # Вычисление градиента между точками
        gradients = []
        for i in range(len(self.known_points)):
            for j in range(i+1, len(self.known_points)):
                dist = self._physical_distance(
                    self.known_points[i], 
                    self.known_points[j]
                )
                if dist < 0.1:  # Только близкие точки
                    value_diff = abs(self.known_values[i] - self.known_values[j])
                    gradient = value_diff / (dist + 1e-10)
                    gradients.append((i, j, gradient))
        
        # Поиск аномальных градиентов
        if not gradients:
            return
        
        grad_values = [g[2] for g in gradients]
        mean_grad = np.mean(grad_values)
        std_grad = np.std(grad_values)
        
        for i, j, grad in gradients:
            if grad > mean_grad + threshold * std_grad:
                # Точка посередине
                mid_point = [
                    (self.known_points[i][k] + self.known_points[j][k]) / 2
                    for k in range(len(self.dim_names))
                ]
                mid_value = (self.known_values[i] + self.known_values[j]) / 2
                
                self.critical_points.append({
                    'point': mid_point,
                    'value': mid_value,
                    'gradient': grad
                })
        
        self.logger.info(f"Found {len(self.critical_points)} critical points")
    
    def add_child_hypercube(self, projection_dims, resolution_factor=0.5):
        """
        Создание дочернего гиперкуба как проекции текущего
        projection_dims: список измерений для проекции
        """
        # Проверка измерений
        for dim in projection_dims:
            if dim not in self.dim_names:
                raise ValueError(f"Dimension {dim} not in hypercube")
        
        # Создание измерений для дочернего гиперкуба
        child_dims = {dim: self.dimensions[dim] for dim in projection_dims}
        
        # Создание дочернего гиперкуба
        child_hypercube = PhysicsHypercubeSystem(
            child_dims,
            resolution=int(self.resolution * resolution_factor),
            parent_hypercube=self
        )
        
        # Перенос известных точек (проекция)
        for point, value in zip(self.known_points, self.known_values):
            projected_point = {
                dim: point[self.dim_names.index(dim)]
                for dim in projection_dims
            }
            child_hypercube.add_known_point(projected_point, value)
        
        self.child_hypercubes.append(child_hypercube)
        self.logger.info(f"Added child hypercube projecting dimensions: {projection_dims}")
        return child_hypercube
    
    def enable_quantum_optimization(self, backend='simulator'):
        """
        Включение квантовой оптимизации для GP модели
        Используются квантовые алгоритмы для обучения модели
        """
        try:
            from qiskit import Aer
            self.quantum_optimization_enabled = True
            self.quantum_backend = Aer.get_backend('qasm_simulator') if backend == 'simulator' else backend
            
            # Перестройка модели с использованием квантовой оптимизации
            if self.gp_model:
                self._build_gaussian_process()
                
            self.logger.info("Quantum optimization enabled")
            return True
        except ImportError:
            self.logger.warning("Quantum libraries not installed. Using classical optimization.")
            return False
    
    def visualize_topology(self):
        """Визуализация топологических инвариантов"""
        if 'betti_curves' not in self.topological_invariants:
            self.calculate_topological_invariants()
            
        if 'betti_curves' not in self.topological_invariants:
            return
            
        betti_curves = self.topological_invariants['betti_curves']
        
        plt.figure(figsize=(10, 6))
        for dim in range(betti_curves.shape[1]):
            plt.plot(betti_curves[0][:, dim], label=f'Betti {dim}')
        
        plt.title('Topological Betti Curves')
        plt.xlabel('Filtration Parameter')
        plt.ylabel('Betti Number')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # ---------------------- КЛЮЧЕВЫЕ УЛУЧШЕНИЯ ИЗ ФАЙЛОВ 1 и 2 ----------------------
    
    def _causal_constraint(self, params):
        """Принцип причинности: будущее не влияет на прошлое"""
        time_dims = [dim for dim in self.dim_names if 'time' in dim.lower()]
        for dim in time_dims:
            if dim in params:
                # Получаем индекс временного измерения
                idx = self.dim_names.index(dim)
                # Все значения этого измерения в известных точках
                all_values = [point[idx] for point in self.known_points]
                if all_values and params[dim] < max(all_values):
                    return False
        return True

    def _deterministic_constraint(self, params):
        """Принцип детерминизма: идентичные причины → идентичные следствия"""
        # Находим точки, близкие к params
        point_vector = [params[dim] for dim in self.dim_names if dim in params]
        similar_values = []
        for known_point in self.known_points:
            # Берем только те измерения, которые есть в params
            known_vector = [known_point[self.dim_names.index(dim)] for dim in params]
            distance = np.linalg.norm(np.array(point_vector) - np.array(known_vector))
            if distance < 0.01:   # порог
                idx = self.known_points.index(known_point)
                similar_values.append(self.known_values[idx])
        
        # Если есть близкие точки, значения должны совпадать
        if similar_values:
            avg_value = np.mean(similar_values)
            std_value = np.std(similar_values)
            if std_value > 0.1 * avg_value:
                return False
        return True

    def set_fundamental_constraint(self, constraint_type):
        """Установка фундаментального ограничения"""
        if constraint_type == 'causality':
            self.physical_constraint = self._causal_constraint
        elif constraint_type == 'determinism':
            self.physical_constraint = self._deterministic_constraint
        else:
            raise ValueError(f"Unknown fundamental constraint: {constraint_type}")
        self.logger.info(f"Applied {constraint_type} fundamental constraint")

    def calculate_universal_limits(self):
        """Расчет фундаментальных пределов системы по Бекенштейну"""
        # Информационный предел Вселенной
        bekenstein_bound = 2.5e43  # бит/м
        universe_volume = 4e80  # м³
        total_bits = bekenstein_bound * universe_volume
        
        # Требуемая память для полного гиперкуба
        dim_sizes = [len(self.dimensions[dim]) if isinstance(self.dimensions[dim], list) 
                    else int((self.dimensions[dim][1]-self.dimensions[dim][0])/0.01)+1 
                    for dim in self.dim_names]
        
        required_memory = np.prod(dim_sizes) * 8 / 1e18  # в эксабайтах
        
        # Соотношение с пределом Бекенштейна
        ratio = required_memory / total_bits
        
        return {
            'bekenstein_bound': total_bits,
            'required_memory': required_memory,
            'ratio_to_limit': ratio,
            'feasible': ratio < 1
        }
    
    def apply_quantum_uncertainty(self):
        """Применение принципа квантовой неопределенности к измерениям"""
        for dim in self.dim_names:
            if 'position' in dim.lower():
                # Находим сопряженную величину (импульс)
                momentum_dim = dim.replace('position', 'momentum')
                if momentum_dim in self.dim_names:
                    # Применяем соотношение неопределенностей
                    for i, point in enumerate(self.known_points):
                        pos_idx = self.dim_names.index(dim)
                        mom_idx = self.dim_names.index(momentum_dim)
                        
                        # Δx * Δp ≥ ħ/2
                        uncertainty = 5.27e-35  # ħ/2 в Дж*с
                        current_uncertainty = abs(point[pos_idx] * point[mom_idx])
                        
                        if current_uncertainty < uncertainty:
                            # Корректируем значения для соблюдения принципа
                            scale = np.sqrt(uncertainty / current_uncertainty)
                            self.known_points[i][pos_idx] *= scale
                            self.known_points[i][mom_idx] *= scale
                    
                    self.logger.info(f"Applied quantum uncertainty to {dim}-{momentum_dim} pair")

    # ---------------------- СУЩЕСТВУЮЩИЕ МЕТОДЫ С МИНИМАЛЬНЫМИ ИЗМЕНЕНИЯМИ ----------------------
    
    def _approximate_line_values(self, line, num_points=10):
        """Аппроксимация значений вдоль коллизионной линии"""
        base = np.array(line['base'])
        direction = np.array(line['direction'])
        
        # Генерируем точки вдоль линии
        t_values = np.linspace(0, 1, num_points)
        points = [base + t * direction for t in t_values]
        
        # Предсказываем значения
        values = []
        for point in points:
            values.append(self._gp_predict(point.tolist()))
        
        line['values'] = list(values)
        self.logger.debug(f"Approximated values for collision line: {values}")
    
    def _project_to_line(self, point, line):
        """Проекция точки на линию и вычисление параметра t"""
        base = np.array(line['base'])
        direction = np.array(line['direction'])
        
        # Вектор от базовой точки до исследуемой
        v = np.array(point) - base
        
        # Длина направления
        dir_length = np.linalg.norm(direction)
        if dir_length < 1e-10:
            return 0, np.inf
            
        # Единичный вектор направления
        unit_dir = direction / dir_length
        
        # Проекция
        t = np.dot(v, unit_dir)
        projection = base + t * unit_dir
        
        # Расстояние до линии
        distance = np.linalg.norm(np.array(point) - projection)
        
        # Нормализованный параметр t (0-1)
        t_norm = t / dir_length
        return t_norm, distance
    
    def _is_within_extrapolation_limit(self, point):
        """Можно ли экстраполировать в данную точку?"""
        if not self.known_points:
            return False
            
        known_points = np.array(self.known_points)
        for i in range(len(self.dim_names)):
            dim_min = np.min(known_points[:, i])
            dim_max = np.max(known_points[:, i])
            dim_range = dim_max - dim_min
            
            if point[i] < dim_min - self.extrapolation_limit * dim_range:
                return False
            if point[i] > dim_max + self.extrapolation_limit * dim_range:
                return False
        return True
    
    def find_collision_line(self, target_value, tolerance=0.1):
        """Поиск ближайшей коллизионной линии к целевому значению"""
        best_line = None
        min_distance = float('inf')
        
        for line in self.collision_lines:
            if line['values'] is None:
                continue
                
            # Находим минимальное расстояние от целевого значения
            values = np.array(line['values'])
            distances = np.abs(values - target_value)
            min_line_distance = np.min(distances)
            
            if min_line_distance < min_distance:
                min_distance = min_line_distance
                best_line = line
        
        if min_distance < tolerance:
            return best_line, min_distance
        return None, min_distance
    
    def optimize_on_line(self, line, target_value):
        """Поиск оптимальной точки на линии для целевого значения"""
        if line['values'] is None or len(line['values']) < 2:
            return None, None
            
        base = np.array(line['base'])
        direction = np.array(line['direction'])
        
        # Параметризация линии
        t_values = np.linspace(0, 1, len(line['values']))
        values = np.array(line['values'])
        
        # Находим t с минимальным расстоянием до целевого значения
        min_idx = np.argmin(np.abs(values - target_value))
        best_t = t_values[min_idx]
        
        # Оптимальная точка
        optimal_point = base + best_t * direction
        param_dict = {dim: optimal_point[i] for i, dim in enumerate(self.dim_names)}
        
        return param_dict, values[min_idx]
    
    def visualize(self, fixed_dims=None, show_uncertainty=False):
        """Визуализация известных точек и коллизионных линий с возможностью показа неопределенности"""
        if len(self.dim_names) < 2:
            print("Визуализация требует минимум 2 измерения")
            return
            
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Фильтрация по фиксированным измерениям
        filtered_points = []
        filtered_values = []
        
        for point, value in zip(self.known_points, self.known_values):
            if fixed_dims:
                skip = False
                for dim, fixed_value in fixed_dims.items():
                    dim_idx = self.dim_names.index(dim)
                    if abs(point[dim_idx] - fixed_value) > 1e-6:
                        skip = True
                        break
                if skip:
                    continue
            filtered_points.append(point)
            filtered_values.append(value)
        
        # Преобразование в массивы
        if filtered_points:
            points_array = np.array(filtered_points)
            x = points_array[:, 0]
            y = points_array[:, 1]
            z = filtered_values
            
            # Известные точки
            ax.scatter(x, y, z, c='blue', s=50, label='Known Points')
            
            # Коллизионные линии
            for line in self.collision_lines:
                if fixed_dims:
                    skip = False
                    for dim, fixed_value in fixed_dims.items():
                        dim_idx = self.dim_names.index(dim)
                        if abs(line['base'][dim_idx] - fixed_value) > 1e-6:
                            skip = True
                            break
                    if skip:
                        continue
                
                base = np.array(line['base'])
                direction = np.array(line['direction'])
                
                # Генерируем точки вдоль линии
                t_values = np.linspace(0, 1, 20)
                line_points = [base + t * direction for t in t_values]
                line_points = np.array(line_points)
                
                # Предсказываем значения
                if line['values'] is None:
                    line_z = [self._gp_predict(p.tolist()) for p in line_points]
                else:
                    # Интерполируем значения
                    orig_t = np.linspace(0, 1, len(line['values']))
                    line_z = np.interp(t_values, orig_t, line['values'])
                
                # Визуализация линии
                ax.plot(
                    line_points[:, 0], 
                    line_points[:, 1], 
                    line_z, 
                    'r-', 
                    linewidth=2,
                    label='Collision Line'
                )
                
                # Визуализация неопределенности
                if show_uncertainty and self.gp_model is not None:
                    uncertainties = []
                    for p in line_points:
                        _, std = self.physical_query_dict(
                            {dim: p[i] for i, dim in enumerate(self.dim_names)},
                            return_std=True
                        )
                        uncertainties.append(std)
                    
                    ax.plot(
                        line_points[:, 0], 
                        line_points[:, 1], 
                        np.array(line_z) + np.array(uncertainties),
                        'r--', 
                        linewidth=1,
                        alpha=0.5
                    )
                    ax.plot(
                        line_points[:, 0], 
                        line_points[:, 1], 
                        np.array(line_z) - np.array(uncertainties),
                        'r--', 
                        linewidth=1,
                        alpha=0.5
                    )
        
        ax.set_xlabel(self.dim_names[0])
        ax.set_ylabel(self.dim_names[1])
        ax.set_zlabel('Physical Law Value')
        ax.set_title('Physics Hypercube System')
        plt.legend()
        plt.show()
    
    def generate_grid(self, return_std=False):
        """Генерация сетки для визуализации (2D) с возможностью возврата неопределенности"""
        if len(self.dim_names) < 2:
            return None, None, None
            
        # Создаем сетку
        x_dim = self.dim_names[0]
        y_dim = self.dim_names[1]
        
        x_min, x_max = self.dimensions[x_dim]
        y_min, y_max = self.dimensions[y_dim]
        
        x_vals = np.linspace(x_min, x_max, self.resolution)
        y_vals = np.linspace(y_min, y_max, self.resolution)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        Z_std = np.zeros_like(X) if return_std else None
        
        # Заполняем значениями
        for i in range(self.resolution):
            for j in range(self.resolution):
                params = {
                    x_dim: X[i, j],
                    y_dim: Y[i, j]
                }
                if return_std:
                    Z[i, j], Z_std[i, j] = self.physical_query_dict(params, return_std=True)
                else:
                    Z[i, j] = self.physical_query_dict(params)
        
        return (X, Y, Z, Z_std) if return_std else (X, Y, Z)
    
    def visualize_surface(self, show_uncertainty=False):
        """Визуализация поверхности предсказанных значений с возможностью показа неопределенности"""
        if len(self.dim_names) < 2:
            print("Визуализация поверхности требует ровно 2 измерения")
            return
            
        if show_uncertainty:
            X, Y, Z, Z_std = self.generate_grid(return_std=True)
        else:
            X, Y, Z = self.generate_grid()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Поверхность
        surf = ax.plot_surface(
            X, Y, Z, 
            cmap='viridis', 
            alpha=0.7,
            edgecolor='none'
        )
        fig.colorbar(surf, ax=ax, label='Physical Law Value')
        
        # Неопределенность
        if show_uncertainty:
            # Верхняя граница неопределенности
            ax.plot_surface(
                X, Y, Z + Z_std, 
                cmap='Reds', 
                alpha=0.3,
                edgecolor='none'
            )
            
            # Нижняя граница неопределенности
            ax.plot_surface(
                X, Y, Z - Z_std, 
                cmap='Blues', 
                alpha=0.3,
                edgecolor='none'
            )
        
        # Известные точки
        points_array = np.array(self.known_points)
        if len(points_array) > 0:
            x = points_array[:, 0]
            y = points_array[:, 1]
            z = self.known_values
            ax.scatter(x, y, z, c='red', s=50, label='Known Points')
        
        ax.set_xlabel(self.dim_names[0])
        ax.set_ylabel(self.dim_names[1])
        ax.set_zlabel('Value')
        ax.set_title('Predicted Physical Laws Surface')
        plt.legend()
        plt.show()
    
    # ========== ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ ДЛЯ ИНТЕГРАЦИИ С OPTIMIZER ==========
    
    def create_optimizer(self):
        """Создание оптимизатора для этого гиперкуба"""
        return TopologicalHypercubeOptimizer(self)
    
    def apply_fundamental_constraints(self, constraint_type='causal'):
        """Применение фундаментальных ограничений через оптимизатор"""
        optimizer = self.create_optimizer()
        optimizer.fundamental_constraint_integration(constraint_type)
    
    def detect_collective_behavior(self, threshold=0.15):
        """Обнаружение коллективных свойств"""
        optimizer = self.create_optimizer()
        return optimizer.collective_behavior_detection(threshold)
    
    def optimize_with_quantum_entanglement(self, depth=3):
        """Квантовая оптимизация через запутывание состояний"""
        optimizer = self.create_optimizer()
        return optimizer.topological_quantum_optimization(depth=depth)

# ===================================================================
# Класс DynamicPhysicsHypercube (Hypercube-X)
# ===================================================================
class DynamicPhysicsHypercube(PhysicsHypercubeSystem):
    """Динамически эволюционирующая система гиперкуба"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topology_engine = TopologyDynamicsEngine(self)
        self.phase_transition_depth = 0
        self.logger = logging.getLogger("DynamicHypercube")
        self.topological_ensemble_interface = TopologicalEnsembleInterface(self)
        
    def add_known_point(self, point, value):
        """Расширенное добавление точки с динамической адаптацией"""
        requires_evolution = self.topology_engine.evaluate_topology_change(point, value)
        
        super().add_known_point(point, value)
        
        if requires_evolution:
            self.handle_phase_transition(point, value)
    
    def handle_phase_transition(self, trigger_point, trigger_value):
        """Обрабатывает фазовый переход, вызванный новой точкой"""
        self.phase_transition_depth += 1
        self.logger.warning(f"Phase transition detected! Depth: {self.phase_transition_depth}")
        
        self.topology_engine.evolve_topology()
        self.find_critical_points()
        
        if self.phase_transition_depth % 3 == 0:
            self.adaptive_dimensionality_shift()
        
        if self.quantum_optimization_enabled:
            self.quantum_model = None
            self._build_gaussian_process()
        
        self.create_topological_snapshot()
    
    def adaptive_dimensionality_shift(self):
        """Адаптивное изменение размерности пространства"""
        betti_1 = self.topological_invariants['betti_numbers'].get(1, 0)
        
        if betti_1 > 5:
            self.logger.info("Activating dimensional compression")
            self.topological_compression(compression_ratio=0.6)
        else:
            self.logger.info("Activating dimensional expansion")
            self.topology_engine.calculate_topology()
    
    def create_topological_snapshot(self):
        """Создает топологический снимок текущего состояния"""
        snapshot = {
            'dimensions': self.dimensions.copy(),
            'topological_invariants': self.topological_invariants.copy(),
            'critical_points': [cp.copy() for cp in self.critical_points],
            'phase_depth': self.phase_transition_depth,
            'timestamp': time.time()
        }
        
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(json.dumps(snapshot).encode())
        
        if not hasattr(self, 'topological_memory'):
            self.topological_memory = []
        self.topological_memory.append(compressed)
        
        self.logger.info("Topological snapshot created")
    
    def restore_topology_state(self, index=-1):
        """Восстанавливает состояние топологии из топологической памяти"""
        if not hasattr(self, 'topological_memory') or not self.topological_memory:
            self.logger.warning("No topological memory available")
            return
        
        dctx = zstd.ZstdDecompressor()
        snapshot = json.loads(dctx.decompress(self.topological_memory[index]).decode())
        
        self.topological_invariants = snapshot['topological_invariants']
        self.critical_points = snapshot['critical_points']
        self.phase_transition_depth = snapshot['phase_depth']
        
        self.topology_engine.initialize_topology()
        
        self.logger.info(f"Topology restored from memory (phase depth: {self.phase_transition_depth})")

# ===================================================================
# Фабрика создания системы Hypercube-X
# ===================================================================
def create_hypercube_x(dimensions, resolution=100):
    """Фабрика для создания Hypercube-X системы"""
    system = DynamicPhysicsHypercube(dimensions, resolution)
    system.topology_engine.initialize_topology()
    optimizer = TopologicalHypercubeOptimizer(system)
    ensemble = TopologicalEnsembleInterface(system)
    
    return {
        'system': system,
        'optimizer': optimizer,
        'ensemble': ensemble
    }

# ===================================================================
# Пример использования системы Hypercube-X
# ===================================================================
if __name__ == "__main__":
    print("="*50)
    print("Инициализация системы Hypercube-X")
    print("="*50)
    
    # Создание ансамбля физических законов
    physics_ensemble = create_hypercube_x({
        'gravity': (1e-11, 1e-8),
        'quantum_scale': (1e-35, 1e-10),
        'time': (0, 1e17)
    })
    
    system = physics_ensemble['system']
    optimizer = physics_ensemble['optimizer']
    ensemble = physics_ensemble['ensemble']
    
    # Добавление точки, вызывающей фазовый переход
    system.add_known_point({'gravity': 1e-35, 'quantum_scale': 1e-18, 'time': 1e10}, 8.2)
    
    # Направленная эволюция к целевой топологии
    optimizer.trigger_targeted_evolution({
        'betti_numbers': {0: 1, 1: 3, 2: 2},
        'quantum_coherence': 0.9
    })
    
    # Создание параллельной системы
    parallel_sys = ensemble.create_parallel_system("strong_gravity", {
        'gravity': {'type': 'scale', 'factor': 1000}
    })
    
    # Сравнение систем
    comparison = ensemble.compare_systems("base", "strong_gravity")
    print(f"Stability ratio: {comparison['stability_ratio']:.2f}")
    
    # Визуализация топологии
    system.visualize_topology()
    
    # Квантовая оптимизация
    optimizer.topological_quantum_optimization()
    
    # Обнаружение коллективных свойств
    emergent_props = optimizer.detect_collective_behavior()
    print(f"Detected emergent properties: {len(emergent_props)}")
    
    # Применение фундаментальных ограничений
    optimizer.fundamental_constraint_integration('causal')
    
    print("\n" + "="*50)
    print("Демонстрация Hypercube-X завершена")
    print("="*50)

# ===================================================================
# Класс DifferentiableTopology (для топологического квантования)
# ===================================================================
class DifferentiableTopology(Function):
    """Реализация дифференцируемой топологии с автоматическим дифференцированием"""
    
    @staticmethod
    def forward(ctx, X: torch.Tensor, homology_dims: List[int] = [0, 1, 2]):
        """
        Прямой проход: вычисление персистентных диаграмм
        X: входные данные формы [batch_size, n_points, n_features]
        """
        ctx.homology_dims = homology_dims
        diagrams = []
        
        # Вычисляем персистентные гомологии для каждого элемента батча
        for batch in X.detach().numpy():
            vr = VietorisRipsPersistence(homology_dimensions=homology_dims)
            diagram = vr.fit_transform([batch])
            diagrams.append(diagram[0])
            
        ctx.save_for_backward(X)
        ctx.diagrams = diagrams
        
        # Возвращаем тензор с информацией о персистенции
        persistence_tensor = torch.zeros((len(diagrams), len(homology_dims)), dtype=torch.float32)
        
        for i, diagram in enumerate(diagrams):
            for j, dim in enumerate(homology_dims):
                if len(diagram[dim]) > 0:
                    persistence = diagram[dim][:,1] - diagram[dim][:,0]
                    persistence_tensor[i,j] = np.mean(persistence)
                    
        return persistence_tensor
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Обратный проход: вычисление градиентов через конечные разности
        """
        X, = ctx.saved_tensors
        grad_input = torch.zeros_like(X)
        epsilon = 1e-5
        
        # Численное вычисление градиента
        for i in range(X.shape[0]):  # По батчам
            for j in range(X.shape[1]):  # По точкам
                for k in range(X.shape[2]):  # По признакам
                    X_plus = X.clone()
                    X_plus[i,j,k] += epsilon
                    
                    X_minus = X.clone()
                    X_minus[i,j,k] -= epsilon
                    
                    # Прямой проход для возмущенных данных
                    persistence_plus = DifferentiableTopology.forward(ctx, X_plus)
                    persistence_minus = DifferentiableTopology.forward(ctx, X_minus)
                    
                    # Конечная разность
                    diff = (persistence_plus - persistence_minus) / (2 * epsilon)
                    
                    # Скалярное произведение с градиентом от следующего слоя
                    grad_input[i,j,k] = torch.sum(grad_output * diff)
                    
        return grad_input, None

# ===================================================================
# Класс TopologicalQuantization (для топологического квантования)
# ===================================================================
class TopologicalQuantization(nn.Module):
    """Реализация топологического квантования H_k(M) ⊗ ℋ_k"""
    
    def __init__(self, betti_numbers: Dict[int, int], n_qubits_per_dim: int = 2):
        super().__init__()
        self.betti_numbers = betti_numbers
        self.n_qubits_per_dim = n_qubits_per_dim
        
        # Инициализация квантовых состояний для каждой размерности
        self.hilbert_spaces = nn.ModuleDict({
            str(dim): QuantumHilbertSpace(count, n_qubits_per_dim)
            for dim, count in betti_numbers.items()
        })
        
        # Матрица смежности для топологических связей
        self.adjacency = self._build_topological_adjacency()
        
    def _build_topological_adjacency(self) -> torch.Tensor:
        """Строит матрицу смежности между пространствами разных размерностей"""
        dims = sorted(list(self.betti_numbers.keys()))
        n_dims = len(dims)
        adj = torch.zeros((n_dims, n_dims))
        
        for i in range(n_dims-1):
            if dims[i+1] - dims[i] == 1:  # Связываем только соседние размерности
                adj[i,i+1] = 1
                adj[i+1,i] = 1
                
        return adj
    
    def forward(self, persistence_diagrams: List[np.ndarray]) -> torch.Tensor:
        """
        Применяет топологическое квантование к персистентным диаграммам
        Возвращает тензор квантовых состояний размерности [n_dims, n_qubits, 2]
        """
        # Вычисляем среднюю персистенцию для каждого класса
        persistence_avg = {}
        for dim, diagram in enumerate(persistence_diagrams):
            if len(diagram) > 0:
                persistences = diagram[:,1] - diagram[:,0]
                persistence_avg[dim] = np.mean(persistences)
            else:
                persistence_avg[dim] = 0.0
                
        # Нормализация
        total_persistence = sum(persistence_avg.values())
        if total_persistence > 0:
            persistence_avg = {k: v/total_persistence for k, v in persistence_avg.items()}
        
        # Генерируем квантовые состояния
        quantum_states = []
        for dim, space in self.hilbert_spaces.items():
            dim_int = int(dim)
            amplitude = persistence_avg.get(dim_int, 0.0)
            states = space(amplitude)
            quantum_states.append(states)
            
        return torch.stack(quantum_states)

# ===================================================================
# Класс QuantumHilbertSpace (для топологического квантования)
# ===================================================================
class QuantumHilbertSpace(nn.Module):
    """Реализация гильбертова пространства ℋ_k для k-мерных классов"""
    
    def __init__(self, n_states: int, n_qubits: int = 2):
        super().__init__()
        self.n_states = n_states
        self.n_qubits = n_qubits
        
        # Параметризованная квантовая схема
        self.circuit = self._build_parameterized_circuit()
        self.backend = Aer.get_backend('statevector_simulator')
        
    def _build_parameterized_circuit(self) -> QuantumCircuit:
        """Строит параметризованную квантовую схему"""
        qc = QuantumCircuit(self.n_qubits)
        params = ParameterVector('θ', length=self.n_qubits*3)
        
        for i in range(self.n_qubits):
            qc.rx(params[i*3], i)
            qc.ry(params[i*3+1], i)
            qc.rz(params[i*3+2], i)
            
        for i in range(self.n_qubits-1):
            qc.cx(i, i+1)
            
        return qc
    
    def forward(self, amplitude: float) -> torch.Tensor:
        """Генерирует квантовое состояние с заданной амплитудой"""
        # Нормализация амплитуды
        norm_amp = np.sqrt(amplitude) if amplitude > 0 else 0.0
        
        # Устанавливаем параметры схемы
        params = np.arcsin(norm_amp) * np.random.randn(self.n_qubits*3)
        bound_circuit = self.circuit.bind_parameters(params)
        
        # Выполняем схему
        result = execute(bound_circuit, self.backend).result()
        statevector = result.get_statevector()
        
        # Преобразуем в тензор PyTorch
        state_tensor = torch.tensor([statevector.real, statevector.imag], dtype=torch.float32)
        
        return state_tensor

# ===================================================================
# Класс TopologicalEnsembleCoherence (для когерентности ансамбля)
# ===================================================================
class TopologicalEnsembleCoherence:
    """Реализация когерентности ансамбля |Ψ⟩ = ∑ c_i |S_i⟩"""
    
    def __init__(self, base_system: 'EnhancedHypercubeX'):
        self.base = base_system
        self.systems = {}
        self.coefficients = {}
        
    def add_system(self, system_id: str, system: 'EnhancedHypercubeX', coefficient: complex):
        """Добавляет систему в суперпозицию"""
        self.systems[system_id] = system
        self.coefficients[system_id] = coefficient
        self._normalize_coefficients()
        
    def _normalize_coefficients(self):
        """Нормализует коэффициенты для сохранения ∑|c_i|² = 1"""
        total = sum(abs(c)**2 for c in self.coefficients.values())
        if total > 0:
            self.coefficients = {k: v/np.sqrt(total) for k, v in self.coefficients.items()}
        
    def measure_observable(self, observable: callable) -> Dict[str, float]:
        """Измеряет наблюдаемую величину в суперпозиции систем"""
        results = {}
        for system_id, system in self.systems.items():
            c = self.coefficients[system_id]
            value = observable(system)
            results[system_id] = {
                'value': value,
                'weighted_value': abs(c)**2 * value,
                'phase': np.angle(c)
            }
            
        # Усреднение с учетом квантовой интерференции
        total = sum(r['weighted_value'] for r in results.values())
        interference = self._calculate_interference(observable)
        
        results['total'] = total + interference
        return results
    
    def _calculate_interference(self, observable: callable) -> float:
        """Вычисляет квантовую интерференцию между системами"""
        if len(self.systems) < 2:
            return 0.0
            
        # Используем квантовые состояния для вычисления перекрытия
        states = []
        for system_id, system in self.systems.items():
            c = self.coefficients[system_id]
            state = system.quantum_state * c
            states.append(state)
            
        # Вычисляем интерференционные члены
        interference = 0.0
        for i in range(len(states)):
            for j in range(i+1, len(states)):
                overlap = torch.sum(states[i] * states[j].conj()).real
                interference += 2 * overlap * observable(self.systems[i]) * observable(self.systems[j])
                
        return interference
