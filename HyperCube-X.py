import numpy as np
import torch
import gpytorch
import networkx as nx
import logging
import json
import zstandard as zstd
import pickle
import time
from collections import deque, defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import basinhopping
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from scipy.interpolate import Rbf
from scipy.stats import entropy
from ripser import ripser
from persim import PersImage
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hypercube_x.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HypercubeX")

class InvalidDimensionError(ValueError):
    """Ошибка, возникающая при отсутствии необходимых измерений"""
    pass

class PhysicalConstraintViolation(ValueError):
    """Ошибка, возникающая при нарушении физических ограничений"""
    pass

class HypercubeEventSystem:
    """Система событий для реализации паттерна Observer"""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
    
    def subscribe(self, event_type: str, callback: Callable):
        """Подписка на событие"""
        self.subscribers[event_type].append(callback)
    
    def notify(self, event_type: str, data: Any):
        """Уведомление подписчиков о событии"""
        for callback in self.subscribers.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in event callback: {str(e)}")

class TopologyManager:
    """Менеджер топологических операций"""
    
    def __init__(self, hypercube: 'PhysicsHypercubeSystem'):
        self.hypercube = hypercube
        self.logger = logging.getLogger("TopologyManager")
    
    def calculate_invariants(self, max_dim: int = 2):
        """Расчет топологических инвариантов системы"""
        if len(self.hypercube.known_points) < 5:
            logger.warning("Not enough points to calculate topological invariants")
            return
        
        try:
            # Подготовка данных
            X = np.array(self.hypercube.known_points)
            
            # Оптимизация для больших наборов данных
            if len(X) > 1000:
                indices = np.random.choice(len(X), 1000, replace=False)
                X_sampled = X[indices]
            else:
                X_sampled = X
            
            # Нормализация с учетом медианы расстояний
            D = cdist(X_sampled, X_sampled)
            med_dist = np.median(D[D > 0])
            if med_dist > 0:
                X_scaled = X_sampled / med_dist
            else:
                X_scaled = X_sampled
            
            # Вычисление персистентной гомологии
            homology_dimensions = list(range(max_dim + 1))
            persistence = VietorisRipsPersistence(
                metric='euclidean',
                max_edge_length=2.0,
                homology_dimensions=homology_dimensions,
                n_jobs=1
            )
            
            diagrams = persistence.fit_transform([X_scaled])[0]
            
            # Сохранение диаграмм
            self.hypercube.topological_invariants['persistence_diagrams'] = {}
            for dim in homology_dimensions:
                self.hypercube.topological_invariants['persistence_diagrams'][dim] = diagrams[dim].tolist()
            
            # Вычисление чисел Бетти
            self.hypercube.topological_invariants['betti_numbers'] = {}
            for dim in homology_dimensions:
                # Число Бетти - это количество точек, где время умирания равно бесконечности
                infinite_points = diagrams[dim][:, 1] == float('inf')
                self.hypercube.topological_invariants['betti_numbers'][dim] = int(np.sum(infinite_points))
            
            logger.info(f"Topological invariants calculated: Betti numbers = {self.hypercube.topological_invariants['betti_numbers']}")
        except Exception as e:
            logger.error(f"Failed to calculate topological invariants: {str(e)}")
    
    def find_symmetries(self, tolerance: float = 0.05):
        """Поиск симметрий в данных гиперкуба"""
        self.hypercube.symmetries = {}
        
        for dim in self.hypercube.dim_names:
            if dim not in self.hypercube.dimensions or isinstance(self.hypercube.dimensions[dim], list):
                continue
                
            # Получение значений по измерению
            values = np.array([p[self.hypercube.dim_names.index(dim)] for p in self.hypercube.known_points])
            if len(values) < 5:
                continue
                
            # Поиск периодичности
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            
            # Вычисление разностей
            diffs = np.diff(sorted_values)
            mean_diff = np.mean(diffs)
            
            # Проверка на периодичность
            if np.std(diffs) / mean_diff < tolerance:
                period = mean_diff
                self.hypercube.symmetries[dim] = {
                    'type': 'periodic',
                    'period': period,
                    'tolerance': tolerance
                }
        
        logger.info(f"Found symmetries: {self.hypercube.symmetries}")
        return self.hypercube.symmetries
    
    def find_critical_points(self, num_points: int = 100):
        """Поиск критических точек (локальных экстремумов и седловых точек) с использованием оптимизации на основе градиента"""
        self.hypercube.topological_invariants['critical_points'] = []
        
        if not self.hypercube.gp_trained or len(self.hypercube.known_points) < 5:
            logger.warning("Not enough data to find critical points")
            return
        
        try:
            # Определение функции для поиска экстремумов
            def objective_function(point):
                params = {dim: point[i] for i, dim in enumerate(self.hypercube.dim_names)}
                return self.hypercube.physical_query_dict(params)
            
            # Генерация начальных точек
            initial_points = []
            for _ in range(num_points):
                point = [
                    np.random.uniform(*self.hypercube.dimensions[dim]) 
                    for dim in self.hypercube.dim_names
                ]
                initial_points.append(point)
            
            # Поиск критических точек через оптимизацию
            for point in initial_points:
                result = basinhopping(
                    objective_function, 
                    x0=point, 
                    niter=5,
                    T=1.0,
                    stepsize=0.1
                )
                
                # Проверка на критическую точку (малый градиент)
                if np.linalg.norm(result.lowest_optimization_result.jac) < 1e-3:
                    # Проверка физических ограничений
                    params = {dim: result.x[i] for i, dim in enumerate(self.hypercube.dim_names)}
                    if self.hypercube.physical_constraint and not self.hypercube.physical_constraint(params):
                        continue
                        
                    self.hypercube.topological_invariants['critical_points'].append({
                        'point': result.x.tolist(),
                        'value': result.fun,
                        'gradient': result.lowest_optimization_result.jac.tolist()
                    })
            
            logger.info(f"Found {len(self.hypercube.topological_invariants['critical_points'])} critical points")
        except Exception as e:
            logger.error(f"Failed to find critical points: {str(e)}")
    
    def detect_bias_in_nonces(self) -> bool:
        """Обнаружение статистических аномалий в nonces (для ECDSA)"""
        try:
            if 'k' not in self.hypercube.dim_names:
                return False
                
            nonces = [p[self.hypercube.dim_names.index('k')] for p in self.hypercube.known_points]
            if len(nonces) < 10:
                return False
                
            # Проверка на равномерность распределения
            hist, _ = np.histogram(nonces, bins=20, density=True)
            entropy_val = entropy(hist)
            max_entropy = np.log(len(hist))
            
            # Если энтропия значительно ниже максимальной, есть предсказуемость
            return entropy_val < 0.7 * max_entropy
        except Exception as e:
            logger.error(f"Failed to detect nonce bias: {str(e)}")
            return False
    
    def detect_isogeny_collisions(self) -> bool:
        """Обнаружение коллизий в пространстве изогений"""
        try:
            # Предполагаем, что изогенные параметры начинаются с индекса 5
            isogeny_indices = [i for i, dim in enumerate(self.hypercube.dim_names) if dim.startswith('e_')]
            if not isogeny_indices:
                return False
                
            # Создаем хэши для изогенных параметров
            hashes = []
            for point in self.hypercube.known_points:
                isogeny_params = tuple(point[i] for i in isogeny_indices)
                hashes.append(hash(isogeny_params))
            
            # Проверка на коллизии
            return len(hashes) != len(set(hashes))
        except Exception as e:
            logger.error(f"Failed to detect isogeny collisions: {str(e)}")
            return False

class QuantumOptimizer:
    """Оптимизатор для квантовых вычислений"""
    
    def __init__(self, hypercube: 'PhysicsHypercubeSystem'):
        self.hypercube = hypercube
        self.logger = logging.getLogger("QuantumOptimizer")
        self.quantum_core = QuantumCircuitCore(n_qubits=8)
    
    def enable_quantum_optimization(self, backend: str = 'simulator'):
        """Включение квантовой оптимизации для GP модели"""
        try:
            self.quantum_core = QuantumCircuitCore(n_qubits=8)
            self.hypercube.quantum_optimization_enabled = True
            logger.info("Quantum optimization enabled with custom quantum circuit implementation")
            return True
        except Exception as e:
            logger.error(f"Failed to enable quantum optimization: {str(e)}")
            return False
    
    def optimize(self, target_properties: Dict[str, Any], num_iterations: int = 10):
        """
        Квантовая оптимизация системы для достижения целевых свойств.
        
        Параметры:
        target_properties: целевые свойства системы
        num_iterations: количество итераций оптимизации
        
        Возвращает:
        Оптимизированная система
        """
        if not self.hypercube.quantum_optimization_enabled:
            if not self.enable_quantum_optimization():
                return None
        
        try:
            # Создание квантовой схемы для оптимизации
            circuit = self.quantum_core.create_entanglement_circuit(
                self._get_topology_vector()
            )
            
            # Квантовая оптимизация
            best_params = None
            best_score = -float('inf')
            
            for i in range(num_iterations):
                # Генерация случайных параметров
                params = np.random.uniform(0, 2 * np.pi, self.quantum_core.n_qubits)
                
                # Применение параметров к схеме
                parametric_circuit = self._apply_quantum_parameters(circuit, params)
                
                # Вычисление состояния
                state = self.quantum_core.execute_circuit(parametric_circuit)
                
                # Оценка соответствия целевым свойствам
                score = self._evaluate_quantum_state(state, target_properties)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            # Применение лучших параметров
            if best_params is not None:
                optimized_circuit = self._apply_quantum_parameters(circuit, best_params)
                optimized_state = self.quantum_core.execute_circuit(optimized_circuit)
                
                # Обновление системы на основе оптимизированного состояния
                self._update_system_from_quantum_state(optimized_state)
                
                logger.info(f"Quantum optimization completed. Best score: {best_score}")
                return self.hypercube
            
            return None
        except Exception as e:
            logger.error(f"Quantum optimization failed: {str(e)}")
            return None
    
    def _get_topology_vector(self) -> np.ndarray:
        """Получение вектора топологических инвариантов для квантовой схемы"""
        if not self.hypercube.topological_invariants.get('betti_numbers'):
            topology_manager = TopologyManager(self.hypercube)
            topology_manager.calculate_invariants()
        
        # Нормализованный вектор чисел Бетти
        betti_values = list(self.hypercube.topological_invariants['betti_numbers'].values())
        if not betti_values:
            return np.zeros(8)
        
        # Нормализация
        max_betti = max(betti_values)
        if max_betti > 0:
            normalized = np.array(betti_values) / max_betti
        else:
            normalized = np.array(betti_values)
        
        # Дополнение до 8 элементов
        if len(normalized) < 8:
            normalized = np.pad(normalized, (0, 8 - len(normalized)), 'constant')
        
        return normalized[:8]
    
    def _apply_quantum_parameters(self, circuit, params: np.ndarray):
        """Применение параметров к квантовой схеме"""
        # Создаем новую схему с примененными параметрами
        parametric_circuit = QuantumCircuitCore(circuit.n_qubits)
        
        # Копируем базовые операции
        for gate in circuit.gates:
            parametric_circuit.gates.append(gate.copy())
        
        # Применяем параметры
        for i, param in enumerate(params):
            parametric_circuit.rx(param, i % circuit.n_qubits)
        
        return parametric_circuit
    
    def _evaluate_quantum_state(self, state: np.ndarray, target_properties: Dict[str, Any]) -> float:
        """Оценка соответствия квантового состояния целевым свойствам"""
        score = 0.0
        
        # Оценка по числам Бетти
        if 'betti_numbers' in target_properties:
            current_betti = self.hypercube.topological_invariants['betti_numbers']
            target_betti = target_properties['betti_numbers']
            
            for dim, target_val in target_betti.items():
                current_val = current_betti.get(int(dim), 0)
                # Нормализованное расстояние
                score -= abs(target_val - current_val) / (target_val + 1)
        
        # Оценка по когерентности
        if 'quantum_coherence' in target_properties:
            coherence = self._measure_quantum_coherence()
            target_coherence = target_properties['quantum_coherence']
            score -= abs(coherence - target_coherence)
        
        return score
    
    def _update_system_from_quantum_state(self, state: np.ndarray):
        """Обновление системы на основе квантового состояния"""
        # Простой пример: обновление топологических инвариантов
        # В реальной системе это будет сложнее
        probabilities = np.abs(state)**2
        probabilities /= np.sum(probabilities)
        
        # Обновление чисел Бетти на основе вероятностей
        for dim in range(min(len(probabilities), 5)):
            self.hypercube.topological_invariants['betti_numbers'][dim] = int(probabilities[dim] * 10)
        
        logger.info("System updated from quantum state")
    
    def _measure_quantum_coherence(self) -> float:
        """Измерение квантовой когерентности в системе"""
        if not self.hypercube.quantum_optimization_enabled:
            return 0.0
        
        try:
            # Создание эталонной схемы с запутанностью
            ref_circuit = QuantumCircuitCore(2)
            ref_circuit.h(0)
            ref_circuit.cx(0, 1)
            
            # Вычисление состояния
            ref_state = self.quantum_core.execute_circuit(ref_circuit)
            
            # Сравнение с текущим состоянием
            current_state = self.quantum_core.execute_circuit(
                self.quantum_core.create_entanglement_circuit(self._get_topology_vector())
            )
            
            # Вычисление верности (fidelity)
            fidelity = np.abs(np.dot(np.conj(ref_state), current_state))**2
            return float(fidelity)
        except Exception as e:
            logger.error(f"Failed to measure quantum coherence: {str(e)}")
            return 0.0

class DataInterpolator:
    """Менеджер интерполяции и аппроксимации данных"""
    
    def __init__(self, hypercube: 'PhysicsHypercubeSystem'):
        self.hypercube = hypercube
        self.logger = logging.getLogger("DataInterpolator")
        self.gp_model = None
        self.gp_likelihood = None
        self.gp_trained = False
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 1.0,
            'max_error': 0.0
        }
    
    def _initialize_gaussian_process(self):
        """Инициализация модели гауссовского процесса"""
        self.gp_model = None
        self.gp_likelihood = None
        self.gp_trained = False
    
    def build_model(self):
        """Построение модели гауссовского процесса на основе известных точек"""
        if len(self.hypercube.known_points) < 2:
            return
        
        try:
            # Преобразование данных в тензоры
            train_x = torch.tensor(self.hypercube.known_points, dtype=torch.float32)
            train_y = torch.tensor(self.hypercube.known_values, dtype=torch.float32)
            
            # Инициализация likelihood и модели
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # Добавление регуляризации через WhiteNoiseKernel
            model = GaussianProcessModel(
                train_x, 
                train_y, 
                likelihood,
                white_noise=0.1
            )
            
            # Обучение модели
            model.train()
            likelihood.train()
            
            # Оптимизация гиперпараметров
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            training_iter = min(50, 100 // max(1, len(self.hypercube.known_points) // 10))
            for i in range(training_iter):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
            
            # Сохранение обученной модели
            self.gp_model = model
            self.gp_likelihood = likelihood
            self.gp_trained = True
            
            logger.info(f"GP model trained with {len(self.hypercube.known_points)} points")
        except Exception as e:
            logger.error(f"Failed to build GP model: {str(e)}")
            self.gp_trained = False
    
    def query(self, params: Dict[str, float]) -> float:
        """
        Запрос физического значения в точке.
        
        Параметры:
        params: словарь значений по измерениям
        
        Возвращает:
        Значение физической величины в указанной точке
        """
        # Проверка структуры точки
        for dim in self.hypercube.dim_names:
            if dim not in params:
                raise InvalidDimensionError(f"Dimension {dim} missing in query")
        
        # Проверка физической реализуемости
        if self.hypercube.physical_constraint and not self.hypercube.physical_constraint(params):
            logger.warning(f"Query {params} violates physical constraints")
            return float('nan')
        
        # Проверка кэша
        params_tuple = tuple(params[dim] for dim in self.hypercube.dim_names)
        if params_tuple in self.hypercube.cache:
            self.hypercube.cache_metadata[params_tuple]['access_count'] += 1
            return self.hypercube.cache[params_tuple]['value']
        
        # Интерполяция/экстраполяция через GP
        if self.gp_trained:
            try:
                query_x = torch.tensor([params_tuple], dtype=torch.float32)
                self.gp_model.eval()
                self.gp_likelihood.eval()
                
                with torch.no_grad():
                    observed_pred = self.gp_likelihood(self.gp_model(query_x))
                    mean = observed_pred.mean.item()
                
                # Кэширование результата
                self._cache_result(params_tuple, mean, is_permanent=False)
                return mean
            except Exception as e:
                logger.error(f"GP prediction failed: {str(e)}")
        
        # Если GP недоступен, используем RBF-интерполяцию
        return self._advanced_interpolation(params_tuple)
    
    def _advanced_interpolation(self, point: Tuple[float, ...]) -> float:
        """RBF-интерполяция на основе ближайших точек"""
        if not self.hypercube.known_points:
            return float('nan')
        
        # Используем RBF-интерполяцию для более точных результатов
        points = np.array(self.hypercube.known_points)
        values = np.array(self.hypercube.known_values)
        
        # Определение функции для RBF
        def rbf_func(*args):
            point_arr = np.array(args)
            # Найти ближайшие точки
            distances = np.sqrt(np.sum((points - point_arr)**2, axis=1))
            k = min(10, len(distances))
            nearest_indices = np.argsort(distances)[:k]
            
            # Взвешенная интерполяция
            weights = 1.0 / (distances[nearest_indices] + 1e-10)
            return np.sum(weights * values[nearest_indices]) / np.sum(weights)
        
        try:
            rbf = Rbf(*points.T, values, function='thin_plate')
            return float(rbf(*point))
        except:
            # Резервный метод - взвешенная интерполяция
            return rbf_func(*point)
    
    def _cache_result(self, params_tuple: Tuple[float, ...], value: float, 
                     gradient: Optional[List[float]] = None, is_permanent: bool = False):
        """Кэширование результата вычисления с опциональным сжатием"""
        # Управление размером кэша
        if len(self.hypercube.cache) >= self.hypercube.max_cache_size:
            # Удаляем наименее важные элементы
            items = list(self.hypercube.cache_metadata.items())
            items.sort(key=lambda x: (not x[1]['is_permanent'], -x[1]['access_count'], x[1]['timestamp']))
            
            # Удаляем 10% кэша
            num_to_remove = max(1, int(0.1 * self.hypercube.max_cache_size))
            for i in range(num_to_remove):
                del self.hypercube.cache[items[i][0]]
                del self.hypercube.cache_metadata[items[i][0]]
        
        # Сжатие данных при необходимости
        compressor = zstd.ZstdCompressor()
        value_data = {
            'value': value,
            'gradient': gradient or [0.0] * len(self.hypercube.dim_names),
            'timestamp': time.time()
        }
        compressed_value = compressor.compress(pickle.dumps(value_data))
        
        # Сохранение статистики сжатия
        self.compression_stats['original_size'] += len(pickle.dumps(value_data))
        self.compression_stats['compressed_size'] += len(compressed_value)
        if self.compression_stats['original_size'] > 0:
            self.compression_stats['compression_ratio'] = (
                self.compression_stats['compressed_size'] / 
                self.compression_stats['original_size']
            )
        
        # Добавляем новый результат
        self.hypercube.cache[params_tuple] = compressed_value
        self.hypercube.cache_metadata[params_tuple] = {
            'is_permanent': is_permanent,
            'access_count': 1,
            'compressed': True
        }
    
    def decompress_result(self, params_tuple: Tuple[float, ...]) -> Dict:
        """Декомпрессия результата из кэша"""
        if params_tuple not in self.hypercube.cache:
            return None
            
        compressed_data = self.hypercube.cache[params_tuple]
        if not self.hypercube.cache_metadata[params_tuple].get('compressed', False):
            return compressed_data
            
        try:
            decompressor = zstd.ZstdDecompressor()
            decompressed_data = decompressor.decompress(compressed_data)
            return pickle.loads(decompressed_data)
        except Exception as e:
            logger.error(f"Failed to decompress cache: {str(e)}")
            return None

class ConstraintValidator:
    """Менеджер проверки физических ограничений"""
    
    def __init__(self, hypercube: 'PhysicsHypercubeSystem'):
        self.hypercube = hypercube
        self.logger = logging.getLogger("ConstraintValidator")
    
    def validate_point(self, params: Dict[str, float]) -> bool:
        """Проверка точки на соответствие физическим ограничениям"""
        # Проверка структуры точки
        for dim in self.hypercube.dim_names:
            if dim not in params:
                raise InvalidDimensionError(f"Dimension {dim} missing in point")
        
        # Проверка диапазонов
        for dim, (min_val, max_val) in self.hypercube.dimensions.items():
            if dim in params:
                if params[dim] < min_val or params[dim] > max_val:
                    logger.warning(f"Point {params} out of range for dimension {dim}")
                    return False
        
        # Проверка физической реализуемости
        if self.hypercube.physical_constraint and not self.hypercube.physical_constraint(params):
            logger.warning(f"Point {params} violates physical constraints")
            return False
        
        return True
    
    def set_physical_constraint(self, constraint_type: str):
        """Установка фундаментального ограничения"""
        if constraint_type == 'causality':
            self.hypercube.physical_constraint = self._causal_constraint
        elif constraint_type == 'determinism':
            self.hypercube.physical_constraint = self._deterministic_constraint
        else:
            raise ValueError(f"Unknown fundamental constraint: {constraint_type}")
        logger.info(f"Applied {constraint_type} fundamental constraint")
    
    def _causal_constraint(self, params: Dict[str, float]) -> bool:
        """Проверка причинностного ограничения"""
        if 'time' not in self.hypercube.dim_names:
            return True
            
        t_idx = self.hypercube.dim_names.index('time')
        current_t = max([p[t_idx] for p in self.hypercube.known_points]) if self.hypercube.known_points else 0
        return params['time'] <= current_t + 1e-6
    
    def _deterministic_constraint(self, params: Dict[str, float]) -> bool:
        """Проверка детерминированности"""
        # В простейшем случае - проверка на коллизии
        for point, _ in zip(self.hypercube.known_points, self.hypercube.known_values):
            distance = 0
            for i, dim in enumerate(self.hypercube.dim_names):
                if dim in params:
                    distance += ((point[i] - params[dim]) / (self.hypercube.dimensions[dim][1] - self.hypercube.dimensions[dim][0])) ** 2
            if np.sqrt(distance) < self.hypercube.collision_tolerance:
                return False
        return True

class PhysicsHypercubeSystem:
    """Базовый класс для физической гиперкубической системы"""
    
    def __init__(self, dimensions: Dict[str, Union[Tuple[float, float], List[float]]], 
                 resolution: int = 100, 
                 extrapolation_limit: float = 0.2,
                 physical_constraint: Optional[Callable[[Dict[str, float]], bool]] = None,
                 collision_tolerance: float = 0.05,
                 uncertainty_slope: float = 0.1,
                 parent_hypercube: Optional['PhysicsHypercubeSystem'] = None):
        """
        Инициализация физической гиперкубической системы.
        
        Параметры:
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
        self.parent_hypercube = parent_hypercube
        self.child_hypercubes = []
        
        # Инициализация данных
        self.known_points = []  # Известные точки (список кортежей)
        self.known_values = []  # Значения в известных точках
        self.known_gradients = []  # Градиенты в известных точках
        
        # Топологические инварианты
        self.topological_invariants = {
            'betti_numbers': {},
            'persistence_diagrams': {},
            'critical_points': []
        }
        self.symmetries = {}
        
        # Квантовые параметры
        self.quantum_optimization_enabled = False
        self.quantum_backend = None
        self.quantum_model = None
        
        # Топологическое представление
        self.topological_compression = False
        self.boundary_representation = []
        
        # Инициализация компонентов
        self.event_system = HypercubeEventSystem()
        self.topology_manager = TopologyManager(self)
        self.quantum_optimizer = QuantumOptimizer(self)
        self.data_interpolator = DataInterpolator(self)
        self.constraint_validator = ConstraintValidator(self)
        
        # Инициализация кэша
        self._initialize_cache()
        
        # Подписка на события
        self.event_system.subscribe('point_added', self._on_point_added)
        
        logger.info(f"PhysicsHypercubeSystem initialized with dimensions: {self.dim_names}")
    
    def _initialize_cache(self):
        """Инициализация умного кэша для эффективного хранения данных"""
        self.cache = {}
        self.cache_metadata = {}
        self.max_cache_size = 10000  # Максимальный размер кэша
        
        # Автонастройка размера кэша в зависимости от доступной памяти
        try:
            import psutil
            total_mem = psutil.virtual_memory().total
            if total_mem < 8e9:  # < 8GB RAM
                self.max_cache_size = 5000
            elif total_mem < 16e9:  # < 16GB RAM
                self.max_cache_size = 10000
            else:  # >= 16GB RAM
                self.max_cache_size = 20000
        except:
            pass
    
    def _on_point_added(self, data):
        """Обработчик события добавления точки"""
        self.topology_manager.calculate_invariants()
        self.data_interpolator.build_model()
    
    def add_known_point(self, point: Dict[str, float], value: float, gradient: Optional[List[float]] = None):
        """
        Добавление известной точки в гиперкуб.
        
        Параметры:
        point: словарь значений по измерениям
        value: значение физической величины
        gradient: градиент в точке (опционально)
        """
        # Проверка структуры точки
        for dim in self.dim_names:
            if dim not in point:
                raise InvalidDimensionError(f"Dimension {dim} missing in point")
        
        # Проверка физической реализуемости
        if not self.constraint_validator.validate_point(point):
            return False
        
        # Преобразование в упорядоченный список
        ordered_point = [point[dim] for dim in self.dim_names]
        
        # Добавление в базу
        self.known_points.append(ordered_point)
        self.known_values.append(value)
        
        # Сохранение градиента, если предоставлен
        if gradient is not None:
            self.known_gradients.append(gradient)
        else:
            self.known_gradients.append([0.0] * len(self.dim_names))
        
        # Кэширование
        params_tuple = tuple(ordered_point)
        self.cache[params_tuple] = value
        self.cache_metadata[params_tuple] = {
            'is_permanent': True,
            'access_count': 1
        }
        
        # Уведомление об изменении
        self.event_system.notify('point_added', {
            'point': ordered_point,
            'value': value,
            'gradient': gradient
        })
        
        logger.info(f"Added known point: {point} = {value}")
        return True
    
    def physical_query_dict(self, params: Dict[str, float]) -> float:
        """
        Запрос физического значения в точке.
        
        Параметры:
        params: словарь значений по измерениям
        
        Возвращает:
        Значение физической величины в указанной точке
        """
        return self.data_interpolator.query(params)
    
    def calculate_topological_invariants(self, max_dim: int = 2):
        """Расчет топологических инвариантов системы"""
        self.topology_manager.calculate_invariants(max_dim)
    
    def find_symmetries(self, tolerance: float = 0.05):
        """Поиск симметрий в данных гиперкуба"""
        return self.topology_manager.find_symmetries(tolerance)
    
    def find_critical_points(self, num_points: int = 100):
        """Поиск критических точек (локальных экстремумов и седловых точек)"""
        self.topology_manager.find_critical_points(num_points)
    
    def calculate_emergent_properties(self):
        """Расчет эмерджентных свойств системы"""
        results = {
            'nonlinearity': 0.0,
            'entropy': 0.0,
            'coherence': 0.0,
            'emergence_metric': 0.0
        }
        
        # 1. Нелинейность: сравнение линейной и нелинейной моделей
        if len(self.known_points) >= 5:
            try:
                X = np.array(self.known_points)
                y = np.array(self.known_values)
                
                # Линейная модель
                linear_error = self._fit_polynomial_model(X, y, degree=1)
                
                # Нелинейная модель
                nonlinear_error = self._fit_polynomial_model(X, y, degree=3)
                
                # Мера нелинейности
                results['nonlinearity'] = max(0.0, min(1.0, 1.0 - nonlinear_error / (linear_error + 1e-10)))
            except:
                pass
        
        # 2. Энтропия Шеннона
        if self.known_values:
            # Нормализация значений для вычисления вероятностей
            min_val, max_val = min(self.known_values), max(self.known_values)
            if min_val < max_val:
                normalized = [(v - min_val) / (max_val - min_val + 1e-10) for v in self.known_values]
                # Дискретизация
                bins = np.histogram(normalized, bins=10)[0]
                bins = bins / np.sum(bins + 1e-10)
                results['entropy'] = entropy(bins)
        
        # 3. Топологическая когерентность
        if self.topological_invariants['betti_numbers']:
            total_betti = sum(self.topological_invariants['betti_numbers'].values())
            if total_betti > 0:
                betti_values = list(self.topological_invariants['betti_numbers'].values())
                results['coherence'] = 1.0 - entropy(betti_values) / np.log(len(betti_values) + 1e-10)
        
        # 4. Общая мера эмерджентности
        results['emergence_metric'] = (
            0.4 * results['nonlinearity'] + 
            0.3 * (1.0 - min(1.0, results['entropy'] / 2.0)) + 
            0.3 * results['coherence']
        )
        
        logger.info(f"Emergent properties calculated: {results}")
        return results
    
    def _fit_polynomial_model(self, X: np.ndarray, y: np.ndarray, degree: int) -> float:
        """Оценка ошибки полиномиальной модели"""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        return mean_squared_error(y, y_pred)
    
    def compress_to_boundary_representation(self, compression_ratio: float = 0.5):
        """
        Сжатие гиперкуба до граничного представления с сохранением топологической структуры.
        
        Параметры:
        compression_ratio: целевое отношение сжатия (0.0-1.0, где 0.0 - максимальное сжатие)
        """
        if not self.known_points:
            return
        
        # Определение критических точек как основы граничного представления
        if not self.topological_invariants['critical_points']:
            self.find_critical_points()
        
        # Сохранение критических точек
        critical_points = self.topological_invariants['critical_points']
        boundary_points = [(cp['point'], cp['value']) for cp in critical_points]
        
        # Добавление точек на границе диапазонов
        for dim_idx, dim in enumerate(self.dim_names):
            min_val, max_val = self.dimensions[dim]
            
            # Точки с минимальным значением по измерению
            min_points = [p for p in self.known_points if abs(p[dim_idx] - min_val) < 0.01 * (max_val - min_val)]
            if min_points:
                for p in min_points[:max(1, int(len(min_points) * compression_ratio))]:
                    idx = self.known_points.index(p)
                    boundary_points.append((p, self.known_values[idx]))
            
            # Точки с максимальным значением по измерению
            max_points = [p for p in self.known_points if abs(p[dim_idx] - max_val) < 0.01 * (max_val - min_val)]
            if max_points:
                for p in max_points[:max(1, int(len(max_points) * compression_ratio))]:
                    idx = self.known_points.index(p)
                    boundary_points.append((p, self.known_values[idx]))
        
        # Сохранение граничного представления
        self.boundary_representation = boundary_points
        self.topological_compression = True
        
        # Удаление оригинальных данных (кроме критических точек)
        self.known_points = [p for p, _ in boundary_points]
        self.known_values = [v for _, v in boundary_points]
        
        # Перестроение моделей
        self.data_interpolator.build_model()
        
        logger.info(f"Hypercube compressed to boundary representation. Compression ratio: {compression_ratio}")
    
    def reconstruct_from_boundary(self, new_points: int = 100):
        """Восстановление данных из граничного представления с использованием топологических инвариантов"""
        if not self.topological_compression:
            logger.warning("Hypercube is not in compressed state")
            return
        
        # Генерация новых точек на основе критических точек и топологии
        reconstructed_points = []
        reconstructed_values = []
        
        if not self.boundary_representation:
            return
        
        # Восстановление через интерполяцию
        for _ in range(new_points):
            # Случайная интерполяция между граничными точками
            idx1, idx2 = np.random.choice(len(self.boundary_representation), 2, replace=False)
            p1, v1 = self.boundary_representation[idx1]
            p2, v2 = self.boundary_representation[idx2]
            
            alpha = np.random.uniform(0, 1)
            new_point = [alpha * p1[i] + (1 - alpha) * p2[i] for i in range(len(p1))]
            
            # Проверка физических ограничений
            params = {dim: new_point[i] for i, dim in enumerate(self.dim_names)}
            if not self.constraint_validator.validate_point(params):
                continue
                
            # Оценка значения
            new_value = alpha * v1 + (1 - alpha) * v2
            
            reconstructed_points.append(new_point)
            reconstructed_values.append(new_value)
        
        # Добавление восстановленных точек
        for point, value in zip(reconstructed_points, reconstructed_values):
            params = {dim: point[i] for i, dim in enumerate(self.dim_names)}
            self.add_known_point(params, value)
        
        logger.info(f"Reconstructed {len(reconstructed_points)} points from boundary representation")
    
    def calculate_universal_limits(self):
        """Расчет фундаментальных пределов системы по Бекенштейну"""
        # Информационный предел Вселенной (бит/м²)
        bekenstein_bound = 2.5e43
        
        # Объем Вселенной (м³)
        universe_volume = 4e80
        
        # Общее количество информации
        total_bits = bekenstein_bound * universe_volume
        
        # Требуемая память для полного гиперкуба
        dim_sizes = []
        for dim in self.dim_names:
            if isinstance(self.dimensions[dim], list):
                dim_sizes.append(len(self.dimensions[dim]))
            else:
                # Оцениваем размер с шагом 1% от диапазона
                min_val, max_val = self.dimensions[dim]
                dim_sizes.append(int((max_val - min_val) / (0.01 * (max_val - min_val + 1e-10))) + 1)
        
        required_memory = np.prod(dim_sizes) * 8 / 1e18  # в эксабайтах
        
        # Соотношение с пределом Бекенштейна
        ratio = required_memory / total_bits if total_bits > 0 else float('inf')
        
        result = {
            'bekenstein_bound': total_bits,
            'required_memory': required_memory,
            'ratio_to_limit': ratio,
            'feasible': ratio < 1
        }
        
        logger.info(f"Universal limits calculated: {result}")
        return result
    
    def add_child_hypercube(self, projection_dims: List[str], resolution_factor: float = 0.5):
        """
        Создание дочернего гиперкуба как проекции текущего.
        
        Параметры:
        projection_dims: список измерений для проекции
        resolution_factor: коэффициент разрешения для дочернего гиперкуба
        
        Возвращает:
        Дочерний гиперкуб
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
            projected_point = {dim: point[self.dim_names.index(dim)] for dim in projection_dims}
            child_hypercube.add_known_point(projected_point, value)
        
        self.child_hypercubes.append(child_hypercube)
        logger.info(f"Added child hypercube projecting dimensions: {projection_dims}")
        
        return child_hypercube
    
    def visualize_topology(self, max_dim: int = 2):
        """Визуализация топологической структуры гиперкуба"""
        if len(self.known_points) < 5:
            logger.warning("Not enough points to visualize topology")
            return
        
        try:
            # Расчет топологических инвариантов
            self.calculate_topological_invariants(max_dim=max_dim)
            
            # Создание графика
            plt.figure(figsize=(12, 10))
            
            # 1. Персистентные диаграммы
            plt.subplot(2, 2, 1)
            for dim in range(max_dim + 1):
                if dim in self.topological_invariants['persistence_diagrams']:
                    dgm = np.array(self.topological_invariants['persistence_diagrams'][dim])
                    if len(dgm) > 0:
                        plt.scatter(dgm[:, 0], dgm[:, 1], label=f'H{dim}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Birth')
            plt.ylabel('Death')
            plt.title('Persistence Diagram')
            plt.legend()
            
            # 2. Кривые Бетти
            plt.subplot(2, 2, 2)
            if self.topological_invariants['persistence_diagrams']:
                max_death = max([max(dgm[:, 1]) for dgm in self.topological_invariants['persistence_diagrams'].values() if len(dgm) > 0], default=1.0)
                epsilons = np.linspace(0, max_death * 1.1, 100)
                
                for dim in range(max_dim + 1):
                    if dim in self.topological_invariants['persistence_diagrams']:
                        dgm = np.array(self.topological_invariants['persistence_diagrams'][dim])
                        betti_curve = [np.sum((dgm[:, 0] <= eps) & (dgm[:, 1] > eps)) for eps in epsilons]
                        plt.plot(epsilons, betti_curve, label=f'H{dim}')
                
                plt.xlabel('Epsilon')
                plt.ylabel('Betti number')
                plt.title('Betti Curves')
                plt.legend()
            
            # 3. Критические точки
            plt.subplot(2, 2, 3)
            if self.known_points:
                X = np.array(self.known_points)
                if X.shape[1] >= 2:
                    plt.scatter(X[:, 0], X[:, 1], c=self.known_values, cmap='viridis')
                    
                    # Отметить критические точки
                    if self.topological_invariants['critical_points']:
                        crit_points = np.array([cp['point'] for cp in self.topological_invariants['critical_points']])
                        plt.scatter(crit_points[:, 0], crit_points[:, 1], c='red', s=100, marker='X', label='Critical points')
                    
                    plt.colorbar(label='Value')
                    plt.xlabel(self.dim_names[0])
                    plt.ylabel(self.dim_names[1])
                    plt.title('Critical Points')
                    plt.legend()
            
            # 4. Эмерджентные свойства
            plt.subplot(2, 2, 4)
            emergent_props = self.calculate_emergent_properties()
            labels = list(emergent_props.keys())
            values = [emergent_props[k] for k in labels]
            
            plt.bar(labels, values, color='skyblue')
            plt.ylim(0, 1.1)
            plt.title('Emergent Properties')
            plt.xticks(rotation=15)
            
            plt.tight_layout()
            plt.savefig('hypercube_topology.png')
            plt.close()
            
            logger.info("Topology visualization saved to 'hypercube_topology.png'")
        except Exception as e:
            logger.error(f"Failed to visualize topology: {str(e)}")
    
    def generate_grid_parallel(self):
        """Генерация сетки для визуализации с использованием параллельных вычислений"""
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
        points = []
        for i in range(self.resolution):
            for j in range(self.resolution):
                point = {
                    self.dim_names[0]: X[i, j],
                    self.dim_names[1]: Y[i, j]
                }
                # Для остальных измерений используем средние значения
                for dim in self.dim_names[2:]:
                    if isinstance(self.dimensions[dim], tuple):
                        point[dim] = (self.dimensions[dim][0] + self.dimensions[dim][1]) / 2
                    else:
                        point[dim] = self.dimensions[dim][len(self.dimensions[dim]) // 2]
                points.append(point)
        
        # Параллельное вычисление значений
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: self.physical_query_dict(p), points))
        
        # Формирование результата
        Z = np.array(results).reshape(self.resolution, self.resolution)
        return X, Y, Z
    
    def visualize_surface(self):
        """Визуализация поверхности физического закона в 3D"""
        if len(self.dim_names) < 2:
            logger.warning("Need at least 2 dimensions for surface visualization")
            return
        
        try:
            X, Y, Z = self.generate_grid_parallel()
            
            if X is None:
                return
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Построение поверхности
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Отображение известных точек
            if self.known_points:
                points_array = np.array(self.known_points)
                if len(self.dim_names) >= 2:
                    x = points_array[:, 0]
                    y = points_array[:, 1]
                    z = self.known_values
                    ax.scatter(x, y, z, c='red', s=50, label='Known Points')
            
            ax.set_xlabel(self.dim_names[0])
            ax.set_ylabel(self.dim_names[1])
            ax.set_zlabel('Physical Law Value')
            ax.set_title('Physics Hypercube System')
            plt.legend()
            plt.savefig('hypercube_surface.png')
            plt.close()
            
            logger.info("Surface visualization saved to 'hypercube_surface.png'")
        except Exception as e:
            logger.error(f"Failed to visualize surface: {str(e)}")
    
    def interactive_visualization(self):
        """Интерактивная 3D-визуализация с использованием Plotly"""
        try:
            import plotly.graph_objects as go
            
            # Генерация сетки
            X, Y, Z = self.generate_grid_parallel()
            
            if X is None or Y is None or Z is None:
                logger.warning("Not enough dimensions for 3D visualization")
                return
            
            # Создание интерактивного графика
            fig = go.Figure(data=[
                go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Viridis',
                    showscale=True
                )
            ])
            
            # Добавление известных точек
            if self.known_points and len(self.known_points[0]) >= 3:
                points_array = np.array(self.known_points)
                fig.add_trace(go.Scatter3d(
                    x=points_array[:, 0],
                    y=points_array[:, 1],
                    z=self.known_values,
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    name='Known Points'
                ))
            
            fig.update_layout(
                title='Physics Hypercube System - Interactive 3D View',
                scene=dict(
                    xaxis_title=self.dim_names[0],
                    yaxis_title=self.dim_names[1],
                    zaxis_title='Value'
                ),
                width=900,
                height=700
            )
            
            fig.write_html('hypercube_interactive.html')
            logger.info("Interactive visualization saved to 'hypercube_interactive.html'")
        except ImportError:
            logger.warning("Plotly not installed. Install with 'pip install plotly' for interactive visualizations.")
        except Exception as e:
            logger.error(f"Failed to create interactive visualization: {str(e)}")
    
    def create_optimizer(self):
        """Создание оптимизатора для этого гиперкуба"""
        return TopologicalHypercubeOptimizer(self)
    
    def create_ensemble_interface(self):
        """Создание интерфейса для работы с ансамблем систем"""
        return TopologicalEnsembleInterface(self)


class GaussianProcessModel(gpytorch.models.ExactGP):
    """Модель гауссовского процесса для физических законов"""
    
    def __init__(self, train_x, train_y, likelihood, white_noise=0.1):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]) +
            gpytorch.kernels.WhiteNoiseKernel(noise_constraint=gpytorch.constraints.Interval(0.0, white_noise))
        )
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class QuantumCircuitCore:
    """Собственная реализация квантовых схем без зависимости от Qiskit"""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates = []
        self.state = np.zeros(2**n_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # Начальное состояние |0>
        self.logger = logging.getLogger("QuantumCircuitCore")
    
    def apply_gate(self, gate_matrix: np.ndarray, target_qubits: List[int]):
        """Применяет квантовый гейт к указанному кубиту(ам)"""
        # Построение полного оператора через тензорное произведение
        full_operator = np.eye(1, dtype=np.complex128)
        
        for i in range(self.n_qubits):
            if i in target_qubits:
                # Вставляем гейт в правильное место
                idx = target_qubits.index(i)
                if idx < len(gate_matrix.shape) - 1:  # Для многокубитных гейтов
                    current_gate = gate_matrix
                else:
                    current_gate = gate_matrix
            else:
                current_gate = np.eye(2, dtype=np.complex128)
            
            full_operator = np.kron(full_operator, current_gate)
        
        # Применение оператора к состоянию
        self.state = full_operator @ self.state
        self.gates.append((gate_matrix.copy(), target_qubits.copy()))
    
    def rx(self, theta: float, qubit: int):
        """Поворот вокруг оси X"""
        gate = np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
        self.apply_gate(gate, [qubit])
    
    def ry(self, theta: float, qubit: int):
        """Поворот вокруг оси Y"""
        gate = np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
        self.apply_gate(gate, [qubit])
    
    def rz(self, theta: float, qubit: int):
        """Поворот вокруг оси Z"""
        gate = np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=np.complex128)
        self.apply_gate(gate, [qubit])
    
    def h(self, qubit: int):
        """Гейт Адамара"""
        gate = np.array([
            [1, 1],
            [1, -1]
        ], dtype=np.complex128) / np.sqrt(2)
        self.apply_gate(gate, [qubit])
    
    def cx(self, control: int, target: int):
        """CNOT гейт"""
        gate = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        self.apply_gate(gate, [control, target])
    
    def create_entanglement_circuit(self, topology_data: np.ndarray):
        """Создает квантовую схему, отражающую топологию системы"""
        circuit = QuantumCircuitCore(self.n_qubits)
        
        # Инициализация кубитов
        for i in range(self.n_qubits):
            circuit.ry(topology_data[i], i)
        
        # Создание запутанности
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i+1)
        
        return circuit
    
    def execute_circuit(self):
        """Выполняет квантовую схему и возвращает состояние"""
        # Состояние уже вычислено в процессе применения гейтов
        return self.state.copy()
    
    def measure(self, num_shots: int = 1024) -> Dict[str, int]:
        """Измерение квантового состояния"""
        probabilities = np.abs(self.state)**2
        outcomes = np.random.choice(
            range(len(probabilities)), 
            size=num_shots, 
            p=probabilities / np.sum(probabilities)
        )
        
        # Подсчет результатов
        counts = {}
        for outcome in outcomes:
            binary = format(outcome, f'0{self.n_qubits}b')
            counts[binary] = counts.get(binary, 0) + 1
        
        return counts
    
    def execute_on_hardware(self, backend_name='ibmq_lima'):
        """Выполнение на реальном квантовом оборудовании через Qiskit"""
        try:
            from qiskit import QuantumCircuit as QiskitCircuit
            from qiskit import Aer, execute
            from qiskit.providers.ibmq import IBMQ
            
            # Конвертация в Qiskit-схему
            qiskit_circuit = QiskitCircuit(self.n_qubits)
            
            # Копирование гейтов
            for gate, qubits in self.gates:
                if gate.shape == (2, 2):  # Однокубитные гейты
                    if np.allclose(gate, np.array([[0, 1], [1, 0]])):
                        qiskit_circuit.x(qubits[0])
                    elif np.allclose(gate, np.array([[1, 0], [0, -1]])):
                        qiskit_circuit.z(qubits[0])
                    elif np.allclose(gate, np.array([[1, 1], [1, -1]]) / np.sqrt(2)):
                        qiskit_circuit.h(qubits[0])
                elif gate.shape == (4, 4):  # Двухкубитные гейты
                    if np.allclose(gate, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])):
                        qiskit_circuit.cx(qubits[0], qubits[1])
            
            # Выполнение на выбранном бэкенде
            if backend_name.startswith('ibmq_'):
                # Попытка подключения к IBM Quantum
                try:
                    IBMQ.load_account()
                    provider = IBMQ.get_provider(hub='ibm-q')
                    backend = provider.get_backend(backend_name)
                except:
                    backend = Aer.get_backend('qasm_simulator')
            else:
                backend = Aer.get_backend(backend_name)
            
            job = execute(qiskit_circuit, backend, shots=1024)
            return job.result().get_counts()
        except ImportError:
            logger.warning("Qiskit not installed. Install with 'pip install qiskit' for hardware execution.")
            return self.measure()
        except Exception as e:
            logger.error(f"Failed to execute on quantum hardware: {str(e)}")
            return self.measure()


class TopologicalEnsembleInterface:
    """Интерфейс для работы с ансамблем систем"""
    
    def __init__(self, base_hypercube: PhysicsHypercubeSystem):
        self.base = base_hypercube
        self.parallel_systems = {}
        self.coefficients = {}
        self.logger = logging.getLogger("TopologicalEnsembleInterface")
    
    def create_parallel_system(self, system_id: str, modification_rules: Dict[str, Dict]):
        """
        Создает параллельную систему с модифицированными законами.
        
        Параметры:
        system_id: идентификатор системы
        modification_rules: правила модификации измерений
        
        Возвращает:
        Новая система
        """
        # Создание копии базовой системы
        new_system = PhysicsHypercubeSystem(
            self.base.dimensions.copy(),
            resolution=self.base.resolution,
            extrapolation_limit=self.base.extrapolation_limit,
            physical_constraint=self.base.physical_constraint,
            collision_tolerance=self.base.collision_tolerance,
            uncertainty_slope=self.base.uncertainty_slope
        )
        
        # Применение модификаций
        for dim, rule in modification_rules.items():
            if dim not in new_system.dimensions:
                continue
                
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
            elif rule['type'] == 'replace':
                new_system.dimensions[dim] = rule['new_range']
        
        # Перенос известных точек
        for point, value in zip(self.base.known_points, self.base.known_values):
            projected_point = {
                dim: point[self.base.dim_names.index(dim)] 
                for dim in self.base.dim_names
                if dim in new_system.dim_names
            }
            new_system.add_known_point(projected_point, value)
        
        self.parallel_systems[system_id] = new_system
        self.coefficients[system_id] = 1.0  # Начальный коэффициент
        
        logger.info(f"Created parallel system '{system_id}' with modifications: {modification_rules}")
        return new_system
    
    def set_coefficient(self, system_id: str, coefficient: complex):
        """Установка коэффициента для системы в ансамбле"""
        if system_id not in self.parallel_systems:
            raise ValueError(f"System {system_id} not found in ensemble")
        
        self.coefficients[system_id] = coefficient
        self._normalize_coefficients()
    
    def _normalize_coefficients(self):
        """Нормализует коэффициенты для сохранения ∑|c_i|² = 1"""
        total = sum(abs(c)**2 for c in self.coefficients.values())
        if total > 0:
            self.coefficients = {k: v/np.sqrt(total) for k, v in self.coefficients.items()}
    
    def measure_observable(self, observable: Callable[[PhysicsHypercubeSystem], float]) -> Dict[str, Any]:
        """
        Измеряет наблюдаемую величину в суперпозиции систем.
        
        Параметры:
        observable: функция, вычисляющая наблюдаемую величину для системы
        
        Возвращает:
        Результаты измерения
        """
        results = {}
        
        # Вычисление для каждой системы
        for system_id, system in self.parallel_systems.items():
            c = self.coefficients[system_id]
            value = observable(system)
            results[system_id] = {
                'value': value,
                'weighted_value': abs(c)**2 * value,
                'phase': np.angle(c)
            }
        
        # Общий результат с учетом интерференции
        total = sum(r['weighted_value'] for r in results.values())
        interference = self._calculate_interference(observable)
        
        results['total'] = total + interference
        results['interference'] = interference
        
        return results
    
    def _calculate_interference(self, observable: Callable[[PhysicsHypercubeSystem], float]) -> float:
        """Вычисляет квантовую интерференцию между системами"""
        if len(self.parallel_systems) < 2:
            return 0.0
        
        # Получение квантовых состояний систем
        states = []
        for system_id, system in self.parallel_systems.items():
            c = self.coefficients[system_id]
            
            # Получение состояния системы (упрощенно)
            if not hasattr(system, 'quantum_state') or system.quantum_state is None:
                # Создаем состояние на основе топологических инвариантов
                betti_values = list(system.topological_invariants['betti_numbers'].values())
                if not betti_values:
                    system.find_critical_points()
                    betti_values = [len(system.topological_invariants['critical_points'])]
                
                # Нормализация
                max_val = max(betti_values) if betti_values else 1
                if max_val > 0:
                    normalized = np.array(betti_values) / max_val
                else:
                    normalized = np.array(betti_values)
                
                # Дополнение до 8 элементов
                if len(normalized) < 8:
                    normalized = np.pad(normalized, (0, 8 - len(normalized)), 'constant')
                
                # Создание квантового состояния
                state_vector = np.zeros(2**3, dtype=np.complex128)  # 3 кубита для простоты
                for i in range(min(len(state_vector), len(normalized))):
                    state_vector[i] = normalized[i] + 0j
                state_vector /= np.linalg.norm(state_vector)
                
                system.quantum_state = state_vector
            
            states.append(system.quantum_state * c)
        
        # Вычисление интерференционных членов
        interference = 0.0
        for i in range(len(states)):
            for j in range(i+1, len(states)):
                # Скалярное произведение для вычисления перекрытия
                overlap = np.sum(states[i] * np.conj(states[j]))
                interference += 2 * np.real(overlap) * observable(self.parallel_systems[list(self.parallel_systems.keys())[i]]) * observable(self.parallel_systems[list(self.parallel_systems.keys())[j]])
        
        return interference
    
    def compare_systems(self, system_id1: str, system_id2: str) -> Dict[str, float]:
        """
        Сравнение двух систем в ансамбле.
        
        Параметры:
        system_id1, system_id2: идентификаторы систем
        
        Возвращает:
        Метрики сравнения
        """
        if system_id1 not in self.parallel_systems or system_id2 not in self.parallel_systems:
            raise ValueError("One or both systems not found in ensemble")
        
        system1 = self.parallel_systems[system_id1]
        system2 = self.parallel_systems[system_id2]
        
        # 1. Сравнение топологии
        topology_diff = self._compare_topology(system1, system2)
        
        # 2. Сравнение критических точек
        critical_points_diff = self._compare_critical_points(system1, system2)
        
        # 3. Сравнение эмерджентных свойств
        emergent_props_diff = self._compare_emergent_properties(system1, system2)
        
        # Общая стабильность
        stability_ratio = 1.0 / (1.0 + topology_diff + critical_points_diff + emergent_props_diff)
        
        return {
            'topology_difference': topology_diff,
            'critical_points_difference': critical_points_diff,
            'emergent_properties_difference': emergent_props_diff,
            'stability_ratio': stability_ratio
        }
    
    def _compare_topology(self, system1: PhysicsHypercubeSystem, system2: PhysicsHypercubeSystem) -> float:
        """Сравнение топологических инвариантов двух систем"""
        if not system1.topological_invariants['betti_numbers'] or not system2.topological_invariants['betti_numbers']:
            system1.calculate_topological_invariants()
            system2.calculate_topological_invariants()
        
        # Сравнение чисел Бетти
        diff = 0.0
        all_dims = set(system1.topological_invariants['betti_numbers'].keys()) | set(system2.topological_invariants['betti_numbers'].keys())
        
        for dim in all_dims:
            val1 = system1.topological_invariants['betti_numbers'].get(dim, 0)
            val2 = system2.topological_invariants['betti_numbers'].get(dim, 0)
            diff += abs(val1 - val2) / (max(val1, val2, 1))
        
        return diff / max(len(all_dims), 1)
    
    def _compare_critical_points(self, system1: PhysicsHypercubeSystem, system2: PhysicsHypercubeSystem) -> float:
        """Сравнение критических точек двух систем"""
        if not system1.topological_invariants['critical_points']:
            system1.find_critical_points()
        if not system2.topological_invariants['critical_points']:
            system2.find_critical_points()
        
        cp1 = system1.topological_invariants['critical_points']
        cp2 = system2.topological_invariants['critical_points']
        
        if not cp1 or not cp2:
            return 1.0  # Максимальное различие
        
        # Вычисление среднего расстояния между критическими точками
        total_dist = 0.0
        count = 0
        
        for p1 in cp1:
            min_dist = float('inf')
            for p2 in cp2:
                dist = np.sqrt(sum((a - b)**2 for a, b in zip(p1['point'], p2['point'])))
                min_dist = min(min_dist, dist)
            total_dist += min_dist
            count += 1
        
        return total_dist / count
    
    def _compare_emergent_properties(self, system1: PhysicsHypercubeSystem, system2: PhysicsHypercubeSystem) -> float:
        """Сравнение эмерджентных свойств двух систем"""
        props1 = system1.calculate_emergent_properties()
        props2 = system2.calculate_emergent_properties()
        
        diff = 0.0
        for key in props1.keys():
            if key in props2:
                diff += abs(props1[key] - props2[key])
        
        return diff / len(props1)


class TopologicalHypercubeOptimizer:
    """Оптимизатор для системы Hypercube-X"""
    
    def __init__(self, hypercube_system: PhysicsHypercubeSystem):
        """
        Инициализация оптимизатора с системой Hypercube-X.
        
        Параметры:
        hypercube_system: экземпляр PhysicsHypercubeSystem
        """
        self.system = hypercube_system
        self.logger = logging.getLogger("TopologicalHypercubeOptimizer")
        self.dimensionality_graph = nx.Graph()
        
        # Автонастройка параметров оптимизации
        self._auto_configure()
        
        logger.info("TopologicalHypercubeOptimizer initialized")
    
    def _auto_configure(self):
        """Автоматическая настройка параметров оптимизации"""
        num_points = len(self.system.known_points)
        
        # Настройка параметров в зависимости от количества данных
        if num_points < 50:
            self.symbolic_regression_generations = 20
            self.quantum_depth = 2
        elif num_points < 200:
            self.symbolic_regression_generations = 40
            self.quantum_depth = 3
        else:
            self.symbolic_regression_generations = 60
            self.quantum_depth = 4
        
        logger.info(f"Auto-configured: symbolic_generations={self.symbolic_regression_generations}, quantum_depth={self.quantum_depth}")
    
    def topological_quantum_optimization(self, depth: int = 3):
        """Топологически обусловленная квантовая оптимизация"""
        # Получение целевых топологических свойств
        target_topology = self._determine_target_topology()
        
        # Квантовая оптимизация
        return self.system.quantum_optimizer.optimize(target_topology, num_iterations=depth)
    
    def _determine_target_topology(self) -> Dict[str, Any]:
        """Определение целевой топологии для оптимизации"""
        # Пример: целевая топология с определенными числами Бетти
        return {
            'betti_numbers': {0: 1, 1: 3, 2: 2},
            'quantum_coherence': 0.9
        }
    
    def detect_collective_behavior(self, threshold: float = 0.15) -> List[Dict]:
        """
        Обнаружение коллективных свойств в системе.
        
        Параметры:
        threshold: порог для определения аномалий
        
        Возвращает:
        Список обнаруженных коллективных свойств
        """
        # Вычисление эмерджентных свойств
        emergent_props = self.system.calculate_emergent_properties()
        
        # Поиск нелинейных взаимодействий
        nonlinear_interactions = self._detect_nonlinear_interactions()
        
        # Поиск критических точек
        self.system.find_critical_points()
        
        # Формирование результатов
        results = []
        
        # 1. Нелинейные взаимодействия
        if nonlinear_interactions > threshold:
            results.append({
                'type': 'nonlinear_interaction',
                'strength': nonlinear_interactions,
                'description': 'Strong nonlinear interactions indicating collective behavior'
            })
        
        # 2. Критические точки
        if len(self.system.topological_invariants['critical_points']) > 5:
            results.append({
                'type': 'critical_points',
                'count': len(self.system.topological_invariants['critical_points']),
                'description': 'Multiple critical points indicating phase transitions'
            })
        
        # 3. Высокая топологическая когерентность
        if emergent_props['coherence'] > 1.0 - threshold:
            results.append({
                'type': 'topological_coherence',
                'value': emergent_props['coherence'],
                'description': 'High topological coherence indicating emergent structure'
            })
        
        return results
    
    def _detect_nonlinear_interactions(self) -> float:
        """Оценка силы нелинейных взаимодействий"""
        # Сравнение линейной и нелинейной моделей
        if len(self.system.known_points) < 5:
            return 0.0
        
        X = np.array(self.system.known_points)
        y = np.array(self.system.known_values)
        
        try:
            # Линейная модель
            linear_error = self.system._fit_polynomial_model(X, y, degree=1)
            
            # Нелинейная модель
            nonlinear_error = self.system._fit_polynomial_model(X, y, degree=3)
            
            # Мера нелинейности
            return max(0.0, min(1.0, 1.0 - nonlinear_error / (linear_error + 1e-10)))
        except:
            return 0.0
    
    def fundamental_constraint_integration(self, constraint_type: str):
        """Интеграция фундаментальных ограничений в систему"""
        self.system.constraint_validator.set_physical_constraint(constraint_type)
        logger.info(f"Integrated fundamental constraint: {constraint_type}")
    
    def trigger_targeted_evolution(self, target_properties: Dict[str, Any]):
        """
        Запуск целевой эволюции системы к указанным свойствам.
        
        Параметры:
        target_properties: целевые свойства системы
        """
        # Создание ансамбля для эволюции
        ensemble = self.system.create_ensemble_interface()
        
        # Создание нескольких параллельных систем с небольшими модификациями
        for i in range(5):
            modification_rules = self._generate_modification_rules()
            system_id = f"evolution_{i}"
            ensemble.create_parallel_system(system_id, modification_rules)
        
        # Оценка систем на соответствие целевым свойствам
        best_system = None
        best_score = -np.inf
        
        for system_id, system in ensemble.parallel_systems.items():
            if system_id.startswith("evolution_"):
                score = self._evaluate_system(system, target_properties)
                if score > best_score:
                    best_score = score
                    best_system = system
        
        # Перенос лучших точек в основную систему
        if best_system:
            for point, value in zip(best_system.known_points, best_system.known_values):
                params = {dim: point[i] for i, dim in enumerate(best_system.dim_names)}
                self.system.add_known_point(params, value)
            
            logger.info(f"Targeted evolution completed. Best system score: {best_score}")
    
    def _generate_modification_rules(self) -> Dict[str, Dict]:
        """Генерация правил модификации для эволюции"""
        rules = {}
        
        for dim in self.system.dim_names:
            if np.random.random() < 0.7:  # 70% шанс модификации измерения
                if np.random.random() < 0.5:
                    # Сдвиг
                    shift = np.random.uniform(-0.1, 0.1) * (self.system.dimensions[dim][1] - self.system.dimensions[dim][0])
                    rules[dim] = {'type': 'shift', 'amount': shift}
                else:
                    # Масштабирование
                    factor = np.random.uniform(0.9, 1.1)
                    rules[dim] = {'type': 'scale', 'factor': factor}
        
        return rules
    
    def _evaluate_system(self, system: PhysicsHypercubeSystem, target_properties: Dict[str, Any]) -> float:
        """Оценка системы на соответствие целевым свойствам"""
        score = 0.0
        
        # Соответствие топологии
        if 'betti_numbers' in target_properties:
            system.calculate_topological_invariants()
            betti_match = 0.0
            for dim, target_val in target_properties['betti_numbers'].items():
                system_val = system.topological_invariants['betti_numbers'].get(int(dim), 0)
                # Нормализованное совпадение
                betti_match += 1.0 - abs(target_val - system_val) / (target_val + 1)
            score += betti_match
        
        # Соответствие симметриям
        if 'symmetry_level' in target_properties:
            system.find_symmetries()
            symmetry_match = 0.0
            for dim, target_sym in target_properties['symmetry_level'].items():
                if dim in system.symmetries:
                    # Простая проверка наличия симметрии
                    symmetry_match += 1.0
            score += symmetry_match * 0.5
        
        # Соответствие когерентности
        if 'quantum_coherence' in target_properties:
            coherence = system._measure_quantum_coherence()
            target_coherence = target_properties['quantum_coherence']
            coherence_match = 1.0 - abs(coherence - target_coherence)
            score += coherence_match * 0.3
        
        return score
    
    def dimensionality_reduction(self, target_dim: int = 2):
        """
        Редукция размерности гиперкуба.
        
        Параметры:
        target_dim: целевая размерность
        
        Возвращает:
        Система с пониженной размерностью
        """
        if len(self.system.dim_names) <= target_dim:
            return self.system
        
        # Подготовка данных
        X = np.array(self.system.known_points)
        if len(X) < 2:
            return self.system
        
        # Выбор метода редукции
        if target_dim == 2:
            if self.system.topology_manager.topology_invariants['betti_numbers'].get(1, 0) > 3:
                reducer = umap.UMAP(n_components=target_dim, 
                                   metric=self.system._physical_distance)
                method = "UMAP"
            else:
                reducer = PCA(n_components=target_dim)
                method = "PCA"
        else:
            reducer = PCA(n_components=target_dim)
            method = "PCA"
        
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
            new_point = point.copy()
            new_point.append(float(reduced_points[i, 0]))  # Добавляем первую компоненту
            new_points.append(new_point)
        
        self.system.known_points = new_points
        
        # Удаление оригинальных измерений (кроме двух)
        if len(self.system.dim_names) > 3:  # Сохраняем два оригинальных измерения
            original_dims = self.system.dim_names[:-1]  # Все, кроме нового
            dims_to_remove = original_dims[2:]  # Удаляем все, кроме первых двух
            
            # Фильтрация измерений
            self.system.dim_names = [dim for dim in self.system.dim_names if dim not in dims_to_remove]
            self.system.dimensions = {dim: rng for dim, rng in self.system.dimensions.items() 
                                     if dim in self.system.dim_names}
            
            # Обновление точек
            dim_indices = [i for i, dim in enumerate(original_dims) if dim not in dims_to_remove]
            self.system.known_points = [
                [point[i] for i in dim_indices] + [point[-1]]  # Добавляем редуцированное измерение
                for point in self.system.known_points
            ]
        
        logger.info(f"Dimensionality reduced to {target_dim} using {method}")
        return self.system
    
    def _physical_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Физическое расстояние с учетом шкал измерений"""
        # Взвешенное евклидово расстояние
        weights = np.array([
            1.0 / max(1e-10, (self.system.dimensions[dim][1] - self.system.dimensions[dim][0]))
            for dim in self.system.dim_names
        ])
        return np.sqrt(np.sum(weights * (x - y)**2))


class ECDSAHypercubeModel:
    """Модель гиперкуба для анализа ECDSA"""
    
    def __init__(self, curve_order: int):
        """
        Инициализация модели гиперкуба ECDSA.
        
        Параметры:
        curve_order: порядок эллиптической кривой
        """
        self.n = curve_order
        self.logger = logging.getLogger("ECDSAHypercubeModel")
        
        # Определение пятимерного гиперкуба
        self.dimensions = {
            'r': (0, self.n),    # x-координата точки R = kG
            's': (0, self.n),    # второй компонент подписи
            'z': (0, self.n),    # хеш-сообщение
            'k': (1, self.n-1),  # случайное число (nonce)
            'd': (1, self.n-1)   # приватный ключ
        }
        
        # Создание физической системы
        self.system = PhysicsHypercubeSystem(self.dimensions)
        
        # Добавление фундаментальных ограничений
        self.system.constraint_validator.set_physical_constraint('causality')
        
        logger.info(f"ECDSA Hypercube model initialized with curve order n={self.n}")
    
    def add_signature(self, r: int, s: int, z: int, d: Optional[int] = None):
        """
        Добавление подписи в гиперкуб.
        
        Параметры:
        r, s, z: компоненты подписи
        d: приватный ключ (опционально)
        """
        # Нормализация значений
        params = {
            'r': r / self.n,
            's': s / self.n,
            'z': z / self.n,
            'k': self._calculate_nonce(r, s, z, d) if d is not None else 0.5,  # Неизвестный nonce
            'd': d / self.n if d is not None else 0.5  # Неизвестный приватный ключ
        }
        
        # Значение для гиперкуба - произвольная функция, здесь используем 1.0 для известных точек
        self.system.add_known_point(params, 1.0)
        
        logger.info(f"Added signature: r={r}, s={s}, z={z}, {'d='+str(d) if d is not None else 'd unknown'}")
    
    def _calculate_nonce(self, r: int, s: int, z: int, d: int) -> float:
        """Расчет nonce k из подписи (если известен приватный ключ)"""
        # s = (z + r*d) * k^(-1) mod n
        # => k = (z + r*d) * s^(-1) mod n
        s_inv = pow(s, -1, self.n)
        k = (z + r * d) * s_inv % self.n
        return k / self.n
    
    def analyze_signature_topology(self):
        """Анализ топологии пространства подписей"""
        self.system.topology_manager.calculate_invariants()
        self.system.topology_manager.find_critical_points()
        
        # Проверка на топологическую структуру тора
        betti = self.system.topological_invariants['betti_numbers']
        is_torus = (betti.get(0, 0) == 1 and betti.get(1, 0) == 2)
        
        logger.info(f"Signature topology analysis: Betti numbers = {betti}, is_torus = {is_torus}")
        return {
            'betti_numbers': betti,
            'is_torus': is_torus,
            'critical_points': len(self.system.topological_invariants['critical_points'])
        }
    
    def detect_nonce_relations(self) -> List[Dict]:
        """
        Обнаружение линейных зависимостей между nonces.
        
        Возвращает:
        Список обнаруженных зависимостей
        """
        if len(self.system.known_points) < 2:
            return []
        
        results = []
        points = np.array(self.system.known_points)
        
        # Проверка пар подписей
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                k_i = points[i, 3]  # Нормализованный nonce
                k_j = points[j, 3]
                
                # Проверка линейной зависимости
                if abs(k_j - k_i) < 0.01:  # Пример простой зависимости
                    results.append({
                        'signatures': (i, j),
                        'relation': 'k_j = k_i',
                        'difference': abs(k_j - k_i)
                    })
                elif abs(k_j - 2*k_i) < 0.01:
                    results.append({
                        'signatures': (i, j),
                        'relation': 'k_j = 2*k_i',
                        'difference': abs(k_j - 2*k_i)
                    })
        
        logger.info(f"Detected {len(results)} nonce relations")
        return results
    
    def visualize_ecdsa_topology(self):
        """Визуализация топологии пространства ECDSA"""
        # Создание 2D проекции (r, z)
        projection_dims = ['r', 'z']
        child_hypercube = self.system.add_child_hypercube(projection_dims)
        
        # Визуализация
        child_hypercube.visualize_surface()
        
        # Дополнительная топологическая визуализация
        self.system.visualize_topology()
        
        logger.info("ECDSA topology visualization completed")
    
    def detect_nonce_bias(self) -> bool:
        """Обнаружение статистических аномалий в nonces"""
        return self.system.topology_manager.detect_bias_in_nonces()
    
    def detect_isogeny_collisions(self) -> bool:
        """Обнаружение коллизий в пространстве изогений"""
        return self.system.topology_manager.detect_isogeny_collisions()


class IsogenyHypercubeModel(ECDSAHypercubeModel):
    """Модель гиперкуба для анализа изогенных криптосистем"""
    
    def __init__(self, curve_order: int, isogeny_dimension: int = 3):
        """
        Инициализация модели гиперкуба для изогенных криптосистем.
        
        Параметры:
        curve_order: порядок эллиптической кривой
        isogeny_dimension: размерность пространства изогений
        """
        super().__init__(curve_order)
        
        # Расширение гиперкуба для изогений
        self.isogeny_dimension = isogeny_dimension
        self.isogeny_params = [f'e_{i}' for i in range(1, isogeny_dimension+1)]
        
        # Обновление размеров гиперкуба
        for param in self.isogeny_params:
            self.dimensions[param] = (0, self.n)
        
        # Пересоздание системы с новыми измерениями
        self.system = PhysicsHypercubeSystem(self.dimensions)
        self.system.constraint_validator.set_physical_constraint('causality')
        
        logger.info(f"Isogeny Hypercube model initialized with dimension {isogeny_dimension}")
    
    def add_isogeny_signature(self, r: int, s: int, z: int, isogeny_params: List[int], d: Optional[int] = None):
        """
        Добавление подписи изогенной криптосистемы.
        
        Параметры:
        r, s, z: компоненты подписи
        isogeny_params: параметры изогении
        d: приватный ключ (опционально)
        """
        if len(isogeny_params) != self.isogeny_dimension:
            raise ValueError(f"Expected {self.isogeny_dimension} isogeny parameters")
        
        # Нормализация значений
        params = {
            'r': r / self.n,
            's': s / self.n,
            'z': z / self.n,
            'k': 0.5,  # Неизвестный nonce
            'd': d / self.n if d is not None else 0.5  # Неизвестный приватный ключ
        }
        
        # Добавление параметров изогении
        for i, param in enumerate(self.isogeny_params):
            params[param] = isogeny_params[i] / self.n
        
        # Добавление в гиперкуб
        self.system.add_known_point(params, 1.0)
        
        logger.info(f"Added isogeny signature with params {isogeny_params}")
    
    def analyze_isogeny_topology(self):
        """Анализ топологии пространства изогенных подписей"""
        self.system.topology_manager.calculate_invariants()
        
        # Проверка на (n-1)-мерный тор
        betti = self.system.topological_invariants['betti_numbers']
        expected_betti = {i: 1 for i in range(self.isogeny_dimension)}
        is_torus = all(betti.get(i, 0) == 1 for i in range(self.isogeny_dimension))
        
        logger.info(f"Isogeny topology analysis: Betti numbers = {betti}, expected torus = {expected_betti}, is_torus = {is_torus}")
        return {
            'betti_numbers': betti,
            'expected_betti': expected_betti,
            'is_torus': is_torus,
            'dimension': self.isogeny_dimension
        }


def create_hypercube_x(dimensions: Dict[str, Union[Tuple[float, float], List[float]]], 
                      resolution: int = 100) -> Dict[str, Any]:
    """
    Фабрика для создания системы Hypercube-X.
    
    Параметры:
    dimensions: словарь измерений и их диапазонов
    resolution: разрешение для визуализации
    
    Возвращает:
    Словарь с компонентами системы
    """
    # Создание основной системы
    system = PhysicsHypercubeSystem(dimensions, resolution)
    
    # Создание оптимизатора
    optimizer = TopologicalHypercubeOptimizer(system)
    
    # Создание интерфейса ансамбля
    ensemble = TopologicalEnsembleInterface(system)
    
    logger.info("Hypercube-X system created successfully")
    return {
        'system': system,
        'optimizer': optimizer,
        'ensemble': ensemble
    }


# Пример использования системы Hypercube-X
if __name__ == "__main__":
    print("="*50)
    print("Инициализация системы Hypercube-X")
    print("="*50)
    
    # Создание системы для анализа ECDSA
    ecda_model = ECDSAHypercubeModel(curve_order=79)
    
    # Добавление примеров подписей
    signatures = [
        {'r': 41, 's': 5, 'z': 12, 'd': 27},
        {'r': 19, 's': 13, 'z': 29, 'd': 27},
        {'r': 55, 's': 31, 'z': 47, 'd': 27}
    ]
    
    for sig in signatures:
        ecda_model.add_signature(**sig)
    
    # Анализ топологии
    topology_analysis = ecda_model.analyze_signature_topology()
    
    # Проверка на статистические аномалии в nonces
    nonce_bias = ecda_model.detect_nonce_bias()
    print(f"Nonce bias detected: {nonce_bias}")
    
    # Визуализация
    ecda_model.visualize_ecdsa_topology()
    
    # Создание системы для физического анализа
    physics_ensemble = create_hypercube_x({
        'gravity': (1e-11, 1e-8),
        'quantum_scale': (1e-35, 1e-10),
        'time': (0, 1e17)
    })
    
    system = physics_ensemble['system']
    optimizer = physics_ensemble['optimizer']
    ensemble = physics_ensemble['ensemble']
    
    # Добавление точки, вызывающей фазовый переход
    system.add_known_point({
        'gravity': 1e-35, 
        'quantum_scale': 1e-18, 
        'time': 1e10
    }, 8.2)
    
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
    if system.quantum_optimizer.enable_quantum_optimization():
        optimizer.topological_quantum_optimization(depth=3)
    
    # Обнаружение коллективных свойств
    emergent_props = optimizer.detect_collective_behavior()
    print(f"Detected emergent properties: {len(emergent_props)}")
    
    # Применение фундаментальных ограничений
    optimizer.fundamental_constraint_integration('causal')
    
    # Интерактивная визуализация
    system.interactive_visualization()
    
    print("="*50)
    print("Демонстрация Hypercube-X завершена")
    print("="*50)
