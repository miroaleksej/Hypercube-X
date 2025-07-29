# Hypercube-X: Топологический и Градиентный Анализ Криптографических Систем

## Временно приостанавливаю работу над всеми проектами в связи с отсутствием заинтересованности и обратной связи с Вашей стороны.

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/02c71162-fda7-48cd-b7c6-53ab8235d196" />

![image](https://github.com/user-attachments/assets/46492ffb-93d1-47ee-b159-fba0e9b8a149)

<h3 align="center">Supported by</h3>
<p align="center">
  <a href="https://quantumfund.org">
    <img src="https://img.shields.io/badge/Quantum_Innovation_Grant-2025-blue" height="30">
  </a>
  <a href="https://opencollective.com/qpp">
    <img src="https://img.shields.io/badge/Open_Collective-Support_Us-green" height="30">
  </a>
</p>

---

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=yourusername.qpp" alt="visitors">
  <img src="https://img.shields.io/github/last-commit/yourusername/qpp?color=blue" alt="last commit">
  <img src="https://img.shields.io/github/stars/yourusername/qpp?style=social" alt="stars">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/Code%20style-black-000000.svg)](https://github.com/psf/black)

**Hypercube-X** — это научно обоснованная система для топологического и градиентного анализа криптографических систем, включая ECDSA и изогенные криптосистемы. Система предоставляет строгий математический фреймворк для обнаружения уязвимостей, анализа топологических свойств и оптимизации криптографических протоколов.

## Основные возможности

- **Пятимерный анализ ECDSA**: Математически строгий анализ пятимерного пространства подписей ECDSA $\mathcal{H} = \mathbb{F}_n^5 = \{(r, s, z, k, d)\}$
- **Топологический анализ**: Вычисление персистентных диаграмм, чисел Бетти и критических точек
- **Градиентный анализ**: Обнаружение линейных зависимостей в nonce и оценка уязвимостей
- **Изогенный анализ**: Обобщение модели на пространство изогений для анализа постквантовых криптосистем
- **Квантовая оптимизация**: Интеграция квантовых вычислений для улучшения анализа
- **Динамическая визуализация**: Интерактивные 3D-визуализации топологических структур

## Математическая основа

Hypercube-X основан на строгих математических принципах:

1. **Пятимерное пространство ECDSA**:
   $$
   s \cdot k \equiv z + r \cdot d \pmod{n}
   $$
   где $r$ - x-координата точки $R = kG$, $s$ - второй компонент подписи, $z$ - хеш-сообщение, $k$ - nonce, $d$ - приватный ключ.

2. **Топологическая структура**: Пространство решений ECDSA имеет структуру тора $\mathbb{T}^2 = \mathbb{S}^1 \times \mathbb{S}^1$.

3. **Градиентная формула**:
   $$
   d \equiv -\frac{\partial r/\partial u_z}{\partial r/\partial u_r} \pmod{n}
   $$
   где $u_r = r/n$, $u_z = z/n$.

4. **Изогенная обобщенная модель**:
   $$
   s \cdot k \equiv z + r \cdot d + \mathcal{I}(\phi) \pmod{n}
   $$
   где $\mathcal{I}(\phi)$ - инвариант, связанный с изогенией $\phi$.

## Установка

### Требования
- Python 3.8+
- pip

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Требования к оборудованию
- Для базового анализа: 4 ГБ ОЗУ, современный процессор
- Для квантовой оптимизации: 8+ ГБ ОЗУ, поддержка CUDA (рекомендуется)
- Для анализа больших наборов данных: 16+ ГБ ОЗУ

## Быстрый старт

### Анализ ECDSA
```python
from hypercube_x.ecdsa import ECDSAHypercubeModel

# Создание модели с порядком кривой n=79
ecdsa_model = ECDSAHypercubeModel(curve_order=79)

# Добавление подписей
signatures = [
    {'r': 41, 's': 5, 'z': 12, 'd': 27},
    {'r': 19, 's': 13, 'z': 29, 'd': 27},
    {'r': 55, 's': 31, 'z': 47, 'd': 27}
]

for sig in signatures:
    ecdsa_model.add_signature(**sig)

# Анализ топологии
topology_analysis = ecdsa_model.analyze_signature_topology()
print(f"Topology analysis: {topology_analysis}")

# Проверка на аномалии в nonce
nonce_bias = ecdsa_model.detect_nonce_bias()
print(f"Nonce bias detected: {nonce_bias}")

# Визуализация
ecdsa_model.visualize_ecdsa_topology()
```

### Анализ изогенных криптосистем
```python
from hypercube_x.isogeny import IsogenyHypercubeModel

# Создание модели с порядком кривой n=79 и размерностью изогении 3
isogeny_model = IsogenyHypercubeModel(curve_order=79, isogeny_dimension=3)

# Добавление подписей с изогенными параметрами
isogeny_model.add_isogeny_signature(
    r=41, s=5, z=12, 
    isogeny_params=[7, -3, 5], 
    d=27
)

# Анализ топологии
isogeny_analysis = isogeny_model.analyze_isogeny_topology()
print(f"Isogeny topology analysis: {isogeny_analysis}")

# Проверка на коллизии
collisions = isogeny_model.detect_isogeny_collisions()
print(f"Isogeny collisions detected: {collisions}")
```

## Основные функции

### Топологический анализ
```python
# Расчет топологических инвариантов
system.calculate_topological_invariants()

# Поиск критических точек
system.find_critical_points()

# Анализ симметрий
system.find_symmetries()
```

### Квантовая оптимизация
```python
# Включение квантовой оптимизации
system.enable_quantum_optimization()

# Целевая эволюция топологии
optimizer.trigger_targeted_evolution({
    'betti_numbers': {0: 1, 1: 3, 2: 2},
    'quantum_coherence': 0.9
})
```

### Обнаружение уязвимостей
```python
# Обнаружение аномалий в nonce
system.detect_nonce_bias()

# Обнаружение линейных зависимостей
system.detect_nonce_relations()

# Обнаружение коллективных свойств
optimizer.detect_collective_behavior()
```

### Визуализация
```python
# Визуализация топологии
system.visualize_topology()

# Визуализация поверхности
system.visualize_surface()

# Интерактивная 3D-визуализация
system.interactive_visualization()
```

## Архитектура

Hypercube-X использует модульную архитектуру с четким разделением ответственности:

```
hypercube_x/
├── core/                  # Ядро системы
│   ├── topology.py        # Топологический анализ
│   ├── quantum.py         # Квантовые вычисления
│   ├── interpolation.py   # Методы интерполяции
│   └── constraints.py     # Физические ограничения
├── crypto/                # Криптографические модули
│   ├── ecdsa.py           # Модель ECDSA
│   └── isogeny.py         # Модель изогенных криптосистем
├── optimizers/            # Оптимизаторы
│   └── hypercube.py       # Топологический оптимизатор
├── utils/                 # Вспомогательные утилиты
│   ├── cache.py           # Умный кэш
│   └── gpu.py             # Менеджер GPU
└── visualization/         # Визуализация
    ├── plotly.py          # Интерактивные графики
    └── matplotlib.py      # Статические графики
```

## Лицензия

Этот проект лицензирован в соответствии с лицензией MIT - подробности см. в файле [LICENSE](LICENSE).

## Цитирование

Если вы используете Hypercube-X в своей научной работе, пожалуйста, цитируйте его следующим образом:

```
@software{hypercube_x_2025,
  author = {Hypercube-X Development Team},
  title = {Hypercube-X: Topological and Gradient Analysis of Cryptographic Systems},
  year = {2025},
  url = {https://github.com/miroaleksej/hypercube-x}
}
```

## Контактная информация

Для вопросов и предложений, пожалуйста, свяжитесь с нами:

- Email: miro-aleksej@yandex.ru
- GitHub Issues: https://github.com/miroaleksej/hypercube-x/issues

## Благодарности

Hypercube-X использует следующие замечательные библиотеки:
- [GPyTorch](https://gpytorch.ai/) для гауссовских процессов
- [giotto-tda](https://giotto.ai/) для топологического анализа данных
- [Qiskit](https://qiskit.org/) для квантовых вычислений
- [UMAP](https://umap-learn.readthedocs.io/) для снижения размерности

Спасибо сообществу за их выдающийся вклад в open-source!
