# API Reference: Hypercube-X

## PhysicsHypercubeSystem

**Ядро системы моделирования физических законов в n-мерном параметрическом пространстве**

### Конструктор
```python
PhysicsHypercubeSystem(
    dimensions: Dict[str, Union[Tuple[float, float], List[str]]],
    resolution: int = 100,
    extrapolation_limit: float = 0.2,
    physical_constraint: Optional[Callable[[Dict[str, float]], bool]] = None,
    collision_tolerance: float = 0.05,
    uncertainty_slope: float = 0.1,
    parent_hypercube: Optional['PhysicsHypercubeSystem'] = None
)
```
- `dimensions`: Словарь измерений и их диапазонов (непрерывные) или категорий (дискретные)
- `resolution`: Разрешение сетки для визуализации
- `extrapolation_limit`: Максимальное относительное отклонение для экстраполяции
- `physical_constraint`: Функция проверки физической реализуемости точки
- `collision_tolerance`: Допуск для коллизионных линий
- `uncertainty_slope`: Коэффициент оценки неопределенности
- `parent_hypercube`: Родительский гиперкуб для иерархии

### Основные методы

#### `add_known_point(point: Dict[str, float], value: float)`
Добавляет известную точку в гиперкуб
- `point`: Словарь значений по измерениям
- `value`: Значение физического закона

#### `physical_query_dict(params: Dict[str, float], return_std: bool = False) -> Union[float, Tuple[float, float]]`
Запрос значения физического закона в точке
- `params`: Точка в параметрическом пространстве
- `return_std`: Возвращать ли оценку неопределенности
- Возвращает: Значение закона или (значение, неопределенность)

#### `calculate_topological_invariants()`
Вычисляет топологические инварианты (числа Бетти, персистентные диаграммы)

#### `find_symmetries(tolerance: float = 0.05) -> Dict[str, Dict]`
Обнаружение симметрий в данных
- `tolerance`: Допустимое отклонение для симметрии
- Возвращает: Словарь обнаруженных симметрий

#### `find_critical_points(threshold: float = 0.2)`
Поиск критических точек (фазовых переходов)
- `threshold`: Порог аномального градиента

#### `compress_to_boundary(compression_ratio: float = 0.8)`
Голографическое сжатие данных до граничного представления
- `compression_ratio`: Доля сохраняемых точек

#### `reconstruct_from_boundary(new_points: int = 100)`
Восстановление данных из граничного представления
- `new_points`: Количество генерируемых точек

#### `visualize_surface(show_uncertainty: bool = False)`
3D визуализация поверхности предсказанных значений
- `show_uncertainty`: Отображать область неопределенности

#### `visualize_topology()`
Визуализация топологических кривых Бетти

---

## DynamicPhysicsHypercube

**Расширенная система с динамической эволюцией топологии (наследует PhysicsHypercubeSystem)**

### Дополнительные методы

#### `handle_phase_transition(trigger_point: Dict[str, float], trigger_value: float)`
Обработка фазового перехода, вызванного новой точкой
- `trigger_point`: Точка, вызвавшая переход
- `trigger_value`: Значение в точке перехода

#### `adaptive_dimensionality_shift()`
Адаптивное изменение размерности пространства на основе топологии

#### `create_holographic_memory()`
Создание сжатого слепка текущего состояния

#### `restore_topology_state(index: int = -1)`
Восстановление состояния топологии из голографической памяти
- `index`: Индекс в массиве памяти (-1 - последнее состояние)

---

## HypercubeXOptimizer

**Оптимизатор системы Hypercube-X с квантово-топологическими методами**

### Конструктор
```python
HypercubeXOptimizer(hypercube_system: PhysicsHypercubeSystem)
```

### Основные методы

#### `topological_dimensionality_reduction(target_dim: int = 3) -> Optional[np.ndarray]`
Топологическая редукция размерности
- `target_dim`: Целевая размерность
- Возвращает: Редуцированные точки

#### `quantum_entanglement_optimization(backend: str = 'simulator', depth: Optional[int] = None) -> bool`
Квантовая оптимизация через запутывание состояний
- `backend`: Квантовый бэкенд ('simulator' или реальное устройство)
- `depth`: Глубина квантовой схемы
- Возвращает: Успешность оптимизации

#### `holographic_boundary_analysis() -> Dict[str, Any]`
Расширенный анализ граничных данных
- Возвращает: Словарь характеристик:
  ```python
  {
      'quantum_connectivity': Dict,
      'multiverse_defects': Dict,
      'quantum_entropy': Dict,
      'multiverse_stability': float
  }
  ```

#### `emergent_property_detection(threshold: float = 0.15) -> List[Dict]`
Обнаружение эмерджентных свойств
- `threshold`: Порог значимости
- Возвращает: Список обнаруженных свойств:
  ```python
  [{
      'type': str,
      'strength': float,
      'description': str
  }]
  ```

#### `multiverse_guided_optimization(target_properties: Dict, num_universes: int = 5) -> Optional[DynamicPhysicsHypercube]`
Оптимизация через создание параллельных вселенных
- `target_properties`: Целевые свойства
- `num_universes`: Количество создаваемых вселенных
- Возвращает: Лучшую вселенную

#### `philosophical_constraint_integration(constraint_type: str = 'causal')`
Интеграция философских ограничений
- `constraint_type`: Тип ограничения ('causal', 'deterministic', 'holographic', 'multiverse_consistent')

---

## MultiverseInterface

**Интерфейс для работы с параллельными вселенными**

### Конструктор
```python
MultiverseInterface(base_hypercube: DynamicPhysicsHypercube)
```

### Основные методы

#### `create_parallel_universe(universe_id: str, modification_rules: Dict[str, Dict]) -> DynamicPhysicsHypercube`
Создание параллельной вселенной с модифицированными законами
- `universe_id`: Идентификатор вселенной
- `modification_rules`: Правила модификации измерений:
  ```python
  {
      'dimension_name': {
          'type': 'shift'|'scale'|'invert',
          'amount': float,  # для shift
          'factor': float   # для scale
      }
  }
  ```
- Возвращает: Экземпляр новой вселенной

#### `compare_universes(universe_id1: str, universe_id2: str) -> Dict[str, Any]`
Сравнение двух параллельных вселенных
- Возвращает:
  ```python
  {
      'betti_difference': Dict[int, int],
      'coherence_difference': float,
      'stability_ratio': float
  }
  ```

---

## SmartCache

**Интеллектуальная система кэширования с TTL и многоуровневым хранением**

### Конструктор
```python
SmartCache(
    max_size: int = 10000,
    ttl_minutes: int = 30,
    cache_dir: str = "cache"
)
```

### Основные методы

#### `set(key: Any, value: Any, is_permanent: bool = False)`
Сохранение значения в кэш
- `key`: Ключ (любой хешируемый объект)
- `value`: Сохраняемое значение
- `is_permanent`: Флаг постоянного хранения

#### `get(key: Any) -> Optional[Any]`
Получение значения из кэша
- `key`: Ключ для поиска
- Возвращает: Значение или None, если отсутствует или просрочено

---

## GPUComputeManager

**Управление вычислительными ресурсами (GPU/CPU)**

### Конструктор
```python
GPUComputeManager(resource_threshold: float = 0.8)
```

### Основные методы

#### `execute(func: Callable, *args, **kwargs) -> Any`
Выполнение функции с контролем ресурсов
- `func`: Функция для выполнения (первый аргумент - torch.device)
- Возвращает: Результат выполнения функции

#### `get_resource_utilization() -> Dict[str, float]`
Возвращает текущую загрузку ресурсов
- Возвращает: `{'cpu': float, 'gpu': float}`

---

## TopologyEvolutionEngine

**Двигатель эволюции топологии системы**

### Основные методы

#### `initialize_topology()`
Инициализация начальной топологии и квантовой схемы

#### `evolve_topology()`
Выполняет эволюцию топологии системы

#### `evaluate_topology_change(new_point: List[float], new_value: float) -> bool`
Оценивает влияние новой точки на топологию
- Возвращает: Требуется ли эволюция топологии

---

## QuantumTopologyCore

**Квантовое ядро для топологических вычислений**

### Конструктор
```python
QuantumTopologyCore(n_qubits: int = 8)
```

### Основные методы

#### `create_entanglement_circuit(topology_data: List[float]) -> QuantumCircuit`
Создает квантовую схему, отражающую топологию системы

#### `calculate_topological_fidelity(circuit1: QuantumCircuit, circuit2: QuantumCircuit) -> float`
Вычисляет топологическую верность между двумя состояниями
- Возвращает: Значение верности [0, 1]

---

## Вспомогательные функции

### `create_hypercube_x(dimensions: Dict, resolution: int = 100) -> Dict`
Фабрика для создания системы Hypercube-X
- Возвращает:
  ```python
  {
      'system': DynamicPhysicsHypercube,
      'optimizer': HypercubeXOptimizer,
      'multiverse': MultiverseInterface
  }
  ```

---

## Философские ограничения (примеры)

### Принцип причинности
```python
def causal_constraint(params):
    time_dims = [dim for dim in self.dim_names if 'time' in dim.lower()]
    for dim in time_dims:
        if dim in params:
            prev_values = [p[self.dim_names.index(dim)] 
                          for p in self.known_points if dim in p]
            if prev_values and params[dim] < max(prev_values):
                return False
    return True
```

### Принцип детерминизма
```python
def deterministic_constraint(params):
    point_vector = [params[dim] for dim in self.dim_names]
    similar_values = []
    for known_point in self.known_points:
        distance = self._physical_distance(point_vector, known_point)
        if distance < 0.01:
            idx = self.known_points.index(known_point)
            similar_values.append(self.known_values[idx])
    
    if similar_values:
        std_value = np.std(similar_values)
        if std_value > 0.1 * np.mean(similar_values):
            return False
    return True
```

## Типовой рабочий процесс

```python
# Инициализация системы
physics_multiverse = create_hypercube_x({
    'energy': (0, 1e3),
    'temperature': (0, 1e4),
    'pressure': (0, 1e6)
})

system = physics_multiverse['system']
optimizer = physics_multiverse['optimizer']

# Добавление экспериментальных данных
system.add_known_point({'energy': 500, 'temperature': 3000, 'pressure': 100000}, 0.75)
system.add_known_point({'energy': 800, 'temperature': 5000, 'pressure': 500000}, 1.25)

# Топологический анализ
system.calculate_topological_invariants()

# Квантовая оптимизация
optimizer.quantum_entanglement_optimization(depth=3)

# Мультиверсное исследование
multiverse = physics_multiverse['multiverse']
multiverse.create_parallel_universe("high_energy", {
    'energy': {'type': 'scale', 'factor': 100}
})

# Анализ эмерджентных свойств
emergent_props = optimizer.emergent_property_detection()
print(f"Detected emergent properties: {emergent_props}")

# Визуализация результатов
system.visualize_surface(show_uncertainty=True)
```
