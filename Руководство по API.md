# 📚 Руководство по API: Physics Hypercube System (PHCS) v.3.0

## 🧱 Основные классы

### 1. PhysicsHypercubeSystem
**Ядро системы** - моделирование физических законов в n-мерном пространстве.

```python
system = PhysicsHypercubeSystem(
    dimensions: dict,
    resolution: int = 100,
    extrapolation_limit: float = 0.2,
    physical_constraint: Callable = None,
    collision_tolerance: float = 0.05,
    uncertainty_slope: float = 0.1
)
```

**Параметры**:
- `dimensions`: Словарь измерений формата `{'dim_name': (min, max)}` или `{'dim_name': ['cat1', 'cat2']}`
- `resolution`: Разрешение для генерации сетки
- `extrapolation_limit`: Предел экстраполяции (в долях диапазона)
- `physical_constraint`: Функция физических ограничений
- `collision_tolerance`: Допустимая погрешность для коллизий
- `uncertainty_slope`: Коэффициент неопределенности

---

### 📌 Основные методы

#### Добавление данных
```python
add_known_point(point: dict, value: float) -> None
```
Добавляет известную точку в систему.
```python
# Пример:
system.add_known_point(
    {'gravitational': 6.67430e-11, 'time': 1.0},
    9.8
)
```

#### Запрос значений
```python
physical_query_dict(params: dict, return_std: bool = False) -> Union[float, Tuple[float, float]]
```
Вычисляет значение физического закона в заданной точке.
```python
# Пример:
value = system.physical_query_dict(
    {'gravitational': 5e-11, 'time': 1.5}
)
value, std = system.physical_query_dict(
    {'gravitational': 5e-11, 'time': 1.5},
    return_std=True
)
```

#### Визуализация
```python
visualize(fixed_dims: dict = None, show_uncertainty: bool = False) -> None
```
3D-визуализация известных точек и коллизионных линий.
```python
# Пример:
system.visualize(
    fixed_dims={'time': 1.0},
    show_uncertainty=True
)
```

```python
visualize_surface(show_uncertainty: bool = False) -> None
```
Визуализация поверхности предсказаний.
```python
# Пример:
system.visualize_surface(show_uncertainty=True)
```

---

### 🌀 Оптимизация и анализ

#### Создание оптимизатора
```python
create_optimizer() -> HypercubeOptimizer
```
Создает экземпляр оптимизатора для системы.
```python
optimizer = system.create_optimizer()
```

#### Квантовая оптимизация
```python
optimize_with_quantum_entanglement(depth: int = 3) -> bool
```
Применяет квантовую оптимизацию через запутанность.
```python
# Пример:
system.optimize_with_quantum_entanglement(depth=2)
```

#### Снижение размерности
```python
topological_dimensionality_reduction(target_dim: int = 3) -> np.ndarray
```
Уменьшает размерность пространства с сохранением топологии.
```python
# Пример:
reduced = system.topological_dimensionality_reduction(target_dim=2)
```

---

### 2. AutopilotPhysicsValidator
**Автоматическая валидация и оптимизация системы**.

```python
validator = AutopilotPhysicsValidator(hypercube_system: PhysicsHypercubeSystem)
```

#### Основные методы:
```python
auto_validate_constants(tolerance: float = 0.01) -> dict
```
Проверяет фундаментальные константы системы.
```python
# Пример:
const_report = validator.auto_validate_constants(tolerance=0.005)
```

```python
auto_validate_laws(num_test_points: int = 100, error_threshold: float = 0.05) -> dict
```
Тестирует известные физические законы.
```python
# Пример:
laws_report = validator.auto_validate_laws(num_test_points=200)
```

```python
auto_optimize_system(optimization_steps: int = 3) -> list
```
Запускает процесс автоматической оптимизации.
```python
# Пример:
optimization_history = validator.auto_optimize_system(steps=5)
```

---

### 🌌 Мультивселенная

#### 3. MultiverseSystem
**Сравнительный анализ параллельных вселенных**.

```python
multiverse = MultiverseSystem(hypercubes: List[PhysicsHypercubeSystem])
```

**Методы**:
```python
cross_universe_query(params: dict) -> dict
```
Выполняет запрос во всех вселенных.
```python
# Пример:
results = multiverse.cross_universe_query(
    {'gravitational': 6.67430e-11}
)
```

```python
visualize_multiverse(fixed_dims: dict = None) -> None
```
Визуализирует все вселенные в параллельных координатах.
```python
# Пример:
multiverse.visualize_multiverse(
    fixed_dims={'time': 1.0}
)
```

---

### ⚛️ Квантовые модули

#### 4. QuantumMemory
**Система квантовой памяти с запутанностью**.

```python
memory = system.enable_quantum_memory()
```

**Методы**:
```python
save_memory(memory_id: str, content: Any, emotion_vector: list) -> str
```
Сохраняет память с квантовым состоянием.
```python
# Пример:
memory.save_memory(
    "grav_law", 
    "F = G*m1*m2/r^2",
    [0.8, 0.2, 0.5]
)
```

```python
recall(memory_id: str, superposition: bool = False) -> dict
```
Восстанавливает память, возможно в суперпозиции.
```python
# Пример:
mem = memory.recall("grav_law", superposition=True)
```

---

### 🧮 Топологические нейросети

#### 5. TopologicalNN
**Нейросети на основе алгебраической топологии**.

```python
nn = system.integrate_topological_nn(homology_dims: list = [0, 1, 2])
```

**Методы**:
```python
train(X: np.ndarray, y: np.ndarray) -> None
```
Обучает модель на тонологических признаках.
```python
# Пример:
nn.train(X_train, y_train)
```

```python
predict(X: np.ndarray) -> np.ndarray
```
Предсказывает значения на новых данных.
```python
# Пример:
predictions = nn.predict(X_test)
```

---

## 💡 Примеры использования

### Создание и оптимизация системы
```python
from PHCS_v_3_0 import PhysicsHypercubeSystem, AutopilotPhysicsValidator

# Инициализация системы
dimensions = {
    'gravitational': (1e-39, 1e-34),
    'electromagnetic': (1e-2, 1e2),
    'time': (0, 10)
}
system = PhysicsHypercubeSystem(dimensions)

# Добавление данных
system.add_known_point({'gravitational': 6.67430e-11, 'time': 1.0}, 9.8)
system.add_known_point({'gravitational': 1e-35, 'time': 2.0}, 0.8)

# Автоматическая оптимизация
validator = AutopilotPhysicsValidator(system)
report = validator.auto_optimize_system(optimization_steps=5)

# Визуализация
system.visualize_surface(show_uncertainty=True)
```

### Работа с мультивселенной
```python
from PHCS_v_3_0 import MultiverseSystem

# Создание параллельных вселенных
universe1 = PhysicsHypercubeSystem({'x': (0, 10), 'y': (0, 10)})
universe2 = PhysicsHypercubeSystem({'x': (0, 5), 'y': (0, 5)})

# Анализ мультивселенной
multiverse = MultiverseSystem([universe1, universe2])
results = multiverse.cross_universe_query({'x': 3.0, 'y': 4.0})
multiverse.visualize_multiverse()
```

### Использование квантовой памяти
```python
# Активация памяти
memory = system.enable_quantum_memory()

# Сохранение воспоминаний
memory.save_memory(
    "quantum_fluctuation", 
    "ΔEΔt ≥ ħ/2", 
    [0.9, 0.1, 0.3]
)

# Восстановление с запутанностью
memory.entangle_with("heisenberg_principle")
mem = memory.recall("quantum_fluctuation", superposition=True)
```

---

## 🛠 Вспомогательные функции

### Генерация сетки
```python
X, Y, Z = system.generate_grid(return_std=False)
```
Возвращает координаты сетки и значения физического закона.

### Поиск критических точек
```python
system.find_critical_points(threshold=0.2)
```
Вычисляет критические точки в пространстве параметров.

### Топологическое сжатие
```python
compressed = system.compress_to_boundary(compression_ratio=0.8)
```
Сжимает систему до топологического описания границы.

---

## 📈 Пример вывода

### Результат валидации законов
```json
{
  "newton_gravity": {
    "mse": 0.0021,
    "r2_score": 0.998,
    "avg_error": 0.012,
    "status": "VALID",
    "num_points": 97
  },
  "coulomb_law": {
    "mse": 0.154,
    "r2_score": 0.782,
    "avg_error": 0.214,
    "status": "INVALID",
    "num_points": 95
  }
}
```

### Оптимизационная история
```json
[
  {
    "step": 0,
    "score": 0.65,
    "actions": ["Initial state"]
  },
  {
    "step": 1,
    "score": 0.82,
    "improvement": 0.17,
    "actions": [
      "Enabled quantum optimization",
      "Applied quantum entanglement optimization"
    ]
  }
]
```

Полная документация доступна в [вики проекта](https://github.com/yourusername/physics-hypercube-system/wiki).
