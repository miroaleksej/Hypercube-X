import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import logging
from PHCS_v.3_0 import PhysicsHypercubeSystem, QuantumMemory, TopologicalNN

class AutopilotPhysicsValidator:
    def __init__(self, hypercube_system):
        """
        Инициализация автопилота для автоматического анализа и валидации физических систем
        
        :param hypercube_system: Экземпляр PhysicsHypercubeSystem
        """
        self.system = hypercube_system
        self.logger = logging.getLogger("Autopilot")
        self._setup_logging()
        
        # Физические константы для автоматической валидации
        self.physical_constants = {
            'gravitational': 6.67430e-11,
            'speed_of_light': 299792458,
            'plancks_constant': 6.62607015e-34,
            'electron_charge': 1.60217662e-19,
            'boltzmann_constant': 1.380649e-23
        }
        
        # Известные физические законы для автоматической проверки
        self.known_physical_laws = {
            'newton_gravity': lambda params: 
                -self.physical_constants['gravitational'] * params['m1'] * params['m2'] / (params['r']**2 + 1e-10),
            
            'coulomb_law': lambda params: 
                8.987551787e9 * params['q1'] * params['q2'] / (params['r']**2 + 1e-10),
            
            'planck_energy': lambda params: 
                self.physical_constants['plancks_constant'] * params['frequency'],
            
            'boltzmann_distribution': lambda params: 
                params['energy'] * np.exp(-params['energy'] / (self.physical_constants['boltzmann_constant'] * params['T']))
        }
        
        self.logger.info("Autopilot Physics Validator initialized")
    
    def _setup_logging(self):
        """Настройка системы логирования"""
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def auto_validate_constants(self, tolerance=0.01):
        """
        Автоматическая валидация фундаментальных констант в системе
        
        :param tolerance: Допустимое отклонение (относительное)
        :return: Словарь с результатами валидации
        """
        results = {}
        
        for const_name, true_value in self.physical_constants.items():
            # Проверяем, существует ли константа в измерениях системы
            if const_name in self.system.dim_names:
                # Получаем диапазон значений измерения
                dim_range = self.system.dimensions[const_name]
                
                # Вычисляем среднее значение в системе
                system_value = np.mean(dim_range)
                
                # Вычисляем относительное отклонение
                deviation = abs(system_value - true_value) / true_value
                
                status = "VALID" if deviation <= tolerance else "INVALID"
                
                results[const_name] = {
                    'true_value': true_value,
                    'system_value': system_value,
                    'deviation': deviation,
                    'status': status
                }
                
                self.logger.info(f"Constant {const_name}: {status} (Deviation: {deviation:.2%})")
        
        return results
    
    def auto_validate_laws(self, num_test_points=100, error_threshold=0.05):
        """
        Автоматическая валидация известных физических законов
        
        :param num_test_points: Количество тестовых точек
        :param error_threshold: Порог ошибки для валидации
        :return: Словарь с результатами валидации
        """
        results = {}
        
        for law_name, law_func in self.known_physical_laws.items():
            # Генерируем случайные точки в области определения системы
            test_points = self._generate_test_points(num_test_points)
            
            # Вычисляем предсказанные и ожидаемые значения
            predicted_values = []
            expected_values = []
            
            for point in test_points:
                try:
                    predicted = self.system.physical_query_dict(point)
                    expected = law_func(point)
                    
                    if not np.isnan(predicted) and not np.isnan(expected):
                        predicted_values.append(predicted)
                        expected_values.append(expected)
                except Exception as e:
                    self.logger.warning(f"Error validating law {law_name}: {str(e)}")
            
            if not predicted_values:
                self.logger.error(f"Validation failed for law {law_name}: no valid points")
                continue
            
            # Вычисляем метрики качества
            mse = mean_squared_error(expected_values, predicted_values)
            r2 = r2_score(expected_values, predicted_values)
            avg_error = np.mean(np.abs(np.array(expected_values) - np.array(predicted_values)))
            
            # Определяем статус валидации
            status = "VALID" if r2 > (1 - error_threshold) else "INVALID"
            
            results[law_name] = {
                'mse': mse,
                'r2_score': r2,
                'avg_error': avg_error,
                'status': status,
                'num_points': len(predicted_values)
            }
            
            self.logger.info(f"Law {law_name}: {status} (R²={r2:.4f}, AvgError={avg_error:.4f})")
            
            # Визуализация результатов
            self._visualize_validation(law_name, expected_values, predicted_values)
        
        return results
    
    def _generate_test_points(self, num_points):
        """
        Генерация случайных тестовых точек в пространстве системы
        
        :param num_points: Количество точек для генерации
        :return: Список точек
        """
        test_points = []
        
        for _ in range(num_points):
            point = {}
            for dim, dim_range in self.system.dimensions.items():
                if isinstance(dim_range, tuple):
                    # Непрерывное измерение
                    point[dim] = np.random.uniform(dim_range[0], dim_range[1])
                elif isinstance(dim_range, list):
                    # Категориальное измерение
                    point[dim] = np.random.choice(dim_range)
            test_points.append(point)
        
        return test_points
    
    def _visualize_validation(self, law_name, expected, predicted):
        """
        Визуализация результатов валидации физического закона
        
        :param law_name: Название закона
        :param expected: Ожидаемые значения
        :param predicted: Предсказанные значения
        """
        plt.figure(figsize=(10, 6))
        
        # График рассеяния
        plt.subplot(1, 2, 1)
        plt.scatter(expected, predicted, alpha=0.6)
        plt.plot([min(expected), max(expected)], [min(expected), max(expected)], 'r--')
        plt.xlabel('Expected Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{law_name} Validation\nR²={r2_score(expected, predicted):.4f}')
        plt.grid(True)
        
        # График ошибок
        plt.subplot(1, 2, 2)
        errors = np.array(expected) - np.array(predicted)
        plt.hist(errors, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{law_name}_validation.png")
        plt.close()
        self.logger.info(f"Validation plot saved as {law_name}_validation.png")
    
    def auto_detect_anomalies(self, sigma_threshold=3):
        """
        Автоматическое обнаружение аномалий в данных системы
        
        :param sigma_threshold: Порог для обнаружения выбросов (в сигмах)
        :return: Список аномальных точек
        """
        anomalies = []
        
        if not self.system.known_points:
            self.logger.warning("No known points for anomaly detection")
            return anomalies
        
        # Преобразуем точки в массив
        points_array = np.array(self.system.known_points)
        values_array = np.array(self.system.known_values)
        
        # Вычисляем статистику по значениям
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        # Ищем аномалии по значениям
        for i, value in enumerate(values_array):
            if abs(value - mean_val) > sigma_threshold * std_val:
                anomalies.append({
                    'type': 'value_anomaly',
                    'point': self.system.known_points[i],
                    'value': value,
                    'z_score': (value - mean_val) / std_val,
                    'description': f"Значение отличается на {sigma_threshold}σ от среднего"
                })
        
        # Ищем пространственные аномалии (если достаточно измерений)
        if points_array.shape[1] >= 2:
            # Используем PCA для уменьшения размерности
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_points = pca.fit_transform(points_array)
            
            # Вычисляем расстояние до центра распределения
            center = np.mean(reduced_points, axis=0)
            distances = np.linalg.norm(reduced_points - center, axis=1)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Ищем пространственные аномалии
            for i, dist in enumerate(distances):
                if dist > mean_dist + sigma_threshold * std_dist:
                    anomalies.append({
                        'type': 'spatial_anomaly',
                        'point': self.system.known_points[i],
                        'value': self.system.known_values[i],
                        'distance_z_score': (dist - mean_dist) / std_dist,
                        'description': "Точка значительно удалена от центра распределения"
                    })
        
        self.logger.info(f"Detected {len(anomalies)} anomalies in the system")
        return anomalies
    
    def auto_optimize_system(self, optimization_steps=3):
        """
        Автоматическая оптимизация системы с использованием всех доступных методов
        
        :param optimization_steps: Количество шагов оптимизации
        :return: История улучшений
        """
        improvement_history = []
        
        # Инициализация базовой оценки
        base_validation = self.auto_validate_laws(num_test_points=50)
        base_score = self._calculate_validation_score(base_validation)
        improvement_history.append({
            'step': 0,
            'score': base_score,
            'actions': ['Initial state']
        })
        
        self.logger.info(f"Starting optimization with base score: {base_score:.4f}")
        
        # Цикл оптимизации
        for step in range(1, optimization_steps + 1):
            actions = []
            current_score = improvement_history[-1]['score']
            
            # 1. Применение квантовой оптимизации
            try:
                if not self.system.quantum_optimization_enabled:
                    self.system.enable_quantum_optimization()
                    actions.append("Enabled quantum optimization")
                
                if self.system.optimize_with_quantum_entanglement(depth=3):
                    actions.append("Applied quantum entanglement optimization")
            except Exception as e:
                self.logger.warning(f"Quantum optimization failed: {str(e)}")
            
            # 2. Снижение размерности
            if len(self.system.dim_names) > 4:
                self.system.topological_dimensionality_reduction(target_dim=3)
                actions.append("Applied topological dimensionality reduction")
            
            # 3. Интеграция философских ограничений
            if step % 2 == 0:
                self.system.apply_philosophical_constraints('causal')
                actions.append("Applied causal philosophical constraints")
            else:
                self.system.apply_philosophical_constraints('holographic')
                actions.append("Applied holographic philosophical constraints")
            
            # 4. Анализ границ
            boundary_analysis = self.system.holographic_boundary_analysis()
            if boundary_analysis.get('defects', {}):
                actions.append("Analyzed holographic boundary defects")
            
            # 5. Активация квантовой памяти
            if not hasattr(self.system, 'quantum_memory') or not self.system.quantum_memory:
                self.system.enable_quantum_memory()
                actions.append("Enabled quantum memory system")
            
            # 6. Интеграция топологических нейросетей
            if not hasattr(self.system, 'topo_nn') or not self.system.topo_nn:
                self.system.integrate_topological_nn()
                actions.append("Integrated topological neural network")
            
            # 7. Обнаружение возникающих свойств
            emergent_props = self.system.detect_emergent_properties()
            if emergent_props:
                actions.append(f"Detected {len(emergent_props)} emergent properties")
            
            # Оценка улучшения
            new_validation = self.auto_validate_laws(num_test_points=50)
            new_score = self._calculate_validation_score(new_validation)
            improvement = new_score - current_score
            
            improvement_history.append({
                'step': step,
                'score': new_score,
                'improvement': improvement,
                'actions': actions
            })
            
            self.logger.info(f"Step {step}: Score={new_score:.4f}, Improvement={improvement:.4f}")
            
            # Визуализация прогресса
            self._visualize_optimization_progress(improvement_history)
            
            # Критерий остановки
            if improvement < 0.01 and step >= 2:
                self.logger.info("Optimization converged")
                break
        
        return improvement_history
    
    def _calculate_validation_score(self, validation_results):
        """
        Вычисление общей оценки качества валидации
        
        :param validation_results: Результаты валидации
        :return: Общая оценка (0-1)
        """
        if not validation_results:
            return 0.0
        
        total_r2 = 0.0
        valid_laws = 0
        
        for result in validation_results.values():
            if result['status'] == 'VALID':
                total_r2 += result['r2_score']
                valid_laws += 1
        
        if valid_laws == 0:
            return 0.0
        
        # Средний R² для валидных законов + бонус за количество
        score = (total_r2 / valid_laws) * (0.7 + 0.3 * valid_laws / len(self.known_physical_laws))
        return min(1.0, score)
    
    def _visualize_optimization_progress(self, history):
        """Визуализация прогресса оптимизации"""
        steps = [h['step'] for h in history]
        scores = [h['score'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, scores, 'o-', markersize=8)
        plt.xlabel('Optimization Step')
        plt.ylabel('Validation Score')
        plt.title('Autopilot Optimization Progress')
        plt.grid(True)
        
        # Аннотации для шагов с улучшением
        for i, h in enumerate(history):
            if i > 0 and h['improvement'] > 0.01:
                plt.annotate(f"+{h['improvement']:.3f}", 
                            (h['step'], h['score']),
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center')
        
        plt.savefig("optimization_progress.png")
        plt.close()
        self.logger.info("Optimization progress plot saved")
    
    def full_autopilot_cycle(self):
        """
        Полный цикл работы автопилота:
        1. Валидация констант
        2. Валидация законов
        3. Обнаружение аномалий
        4. Автоматическая оптимизация
        """
        self.logger.info("Starting full autopilot cycle")
        
        # Шаг 1: Валидация фундаментальных констант
        const_results = self.auto_validate_constants()
        num_valid_consts = sum(1 for r in const_results.values() if r['status'] == 'VALID')
        self.logger.info(f"Constants validation: {num_valid_consts}/{len(const_results)} valid")
        
        # Шаг 2: Валидация физических законов
        law_results = self.auto_validate_laws()
        num_valid_laws = sum(1 for r in law_results.values() if r['status'] == 'VALID')
        self.logger.info(f"Laws validation: {num_valid_laws}/{len(law_results)} valid")
        
        # Шаг 3: Обнаружение аномалий
        anomalies = self.auto_detect_anomalies()
        if anomalies:
            self.logger.warning(f"Detected {len(anomalies)} anomalies in the system")
        
        # Шаг 4: Автоматическая оптимизация
        optimization_history = self.auto_optimize_system()
        final_score = optimization_history[-1]['score']
        
        # Итоговый отчет
        report = {
            'constants_validation': const_results,
            'laws_validation': law_results,
            'anomalies_detected': anomalies,
            'optimization_history': optimization_history,
            'final_score': final_score
        }
        
        self.logger.info(f"Autopilot cycle completed. Final validation score: {final_score:.4f}")
        return report

# Пример использования
if __name__ == "__main__":
    # Создание тестовой физической системы
    dimensions = {
        'gravitational': (1e-39, 1e-34),
        'electromagnetic': (1e-2, 1e2),
        'time': (0, 10),
        'mass': (1e-30, 1e30),
        'distance': (1e-15, 1e20)
    }
    
    system = PhysicsHypercubeSystem(dimensions)
    
    # Добавление начальных точек данных
    system.add_known_point({
        'gravitational': 6.67430e-11,
        'electromagnetic': 1/137,
        'time': 1.0,
        'mass': 5.97e24,
        'distance': 6.371e6
    }, 9.8)
    
    # Инициализация автопилота
    autopilot = AutopilotPhysicsValidator(system)
    
    # Запуск полного цикла
    report = autopilot.full_autopilot_cycle()
    
    print("\nFinal Autopilot Report:")
    print(f"Validation Score: {report['final_score']:.4f}")
    print(f"Valid Laws: {sum(1 for r in report['laws_validation'].values() if r['status'] == 'VALID')}/{len(report['laws_validation'])}")
    print(f"Detected Anomalies: {len(report['anomalies_detected'])}")