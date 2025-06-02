# Tutorial Didático: Controle PID com LaneNet no CARLA

Este tutorial explica passo a passo o funcionamento e adaptação do script `carla_setup_pid.py`, que utiliza um modelo de segmentação (LaneNet) para detectar faixas de rodagem no simulador CARLA, e aplica controle PID para ajustar a direção de um veículo.

---

## 📦 Pré-requisitos

* CARLA Simulator (versão compatível)
* Python 3.8+
* PyTorch
* OpenCV
* LaneNet treinado (segmentação binária)

---

## 🎯 Objetivo do Script

* Capturar imagens da câmera frontal do veículo no CARLA.
* Utilizar a LaneNet para segmentar as faixas da pista.
* Calcular o erro lateral entre o centro da imagem e o centro da faixa.
* Aplicar controle PID para ajustar a direção do veículo.
* Exibir visualmente o resultado.

---

## 🧠 Estrutura do Script

### 1. **Carregamento do Modelo**

```python
model = LaneNet().to(device)
checkpoint = torch.load("models/best_7.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

* **`LaneNet()`**: instância do modelo de segmentação binária.
* **`checkpoint`**: carrega os pesos treinados.
* **`.eval()`**: define o modelo no modo de inferência.

### 2. **Configuração do Controle PID**

```python
erro_integral = 0.0
erro_anterior = 0.0
Kp, Ki, Kd = 1.5, 0.1, 0.2
```

* **Kp**: ganho proporcional.
* **Ki**: ganho integral.
* **Kd**: ganho derivativo.

Esses parâmetros controlam o quão agressivo o veículo reage ao desvio da faixa.

### 3. **Função `process_image(image)`**

#### Protótipo

```python
def process_image(image: carla.Image) -> None
```

#### Responsabilidades:

* Convertendo imagem de `carla.Image` para `numpy.ndarray`.
* Segmentar a faixa usando LaneNet.
* Calcular erro lateral.
* Aplicar PID e definir direção (`steer`).
* Exibir a imagem com sobreposição da segmentação e centro da faixa.

#### Principais etapas:

```python
frame_tensor = transforms.ToTensor()(frame_resized).unsqueeze(0).to(device)
frame_tensor = transforms.functional.normalize(frame_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

* Normaliza a imagem para o modelo LaneNet.

```python
raw_output = model(frame_tensor)
lane_mask = (torch.sigmoid(raw_output).squeeze() > 0.9).cpu().numpy().astype(np.uint8)
```

* Aplica o modelo e converte para uma máscara binária.

```python
erro = centro_imagem - centro_faixa
erro_cm = erro * 40.0 / (512 / 2)
```

* Converte o erro de pixels para centímetros.

```python
controle = Kp * erro_cm + Ki * erro_integral + Kd * erro_derivada
```

* Calcula a saída do PID.

### 4. **Função `calculate_steering()`** *(versão adaptada para PID)*

#### Protótipo

```python
def calculate_steering(lane_mask: np.ndarray, image_width: int) -> Tuple[float, float]
```

* Calcula o centro da faixa e erro lateral.
* Retorna a direção corrigida pelo PID.

### 5. **Função `start_simulation()`**

#### Protótipo

```python
def start_simulation() -> None
```

* Conecta ao simulador CARLA.
* Cria o veículo e sensor de câmera RGB.
* Inicia o loop principal, chamando `process_image()` a cada frame.

### 6. **Função `show_images()`**

* Exibe as imagens de forma contínua.
* Permite sair com tecla `q`.

---

## 🖼️ Visualização Gráfica

A figura abaixo mostra:

| Imagem Original                           | Máscara Binária | Sobreposição Final |
| ----------------------------------------- | --------------- | ------------------ |
| ![original](tutorial_segmentacao_pid.png) |                 |                    |

* A linha azul representa o centro da faixa.
* A linha verde representa o centro da imagem.

---

## 🧪 Experimentos Sugeridos

* **Ajuste de PID**: altere os valores de `Kp`, `Ki`, `Kd` para observar diferentes comportamentos de correção.
* **Teste em diferentes pistas**: experimente com diferentes mapas do CARLA.
* **Adição de ruído visual**: para testar a robustez do controle e da segmentação.

---

## 🛠️ Arquivos Envolvidos

* `carla_setup_pid.py`: script principal com controle PID.
* `LaneNet` (model.py): modelo de segmentação binária.
* `models/best_7.pth`: pesos treinados da LaneNet.

---

## ✅ Conclusão

Com este tutorial, você entende como usar segmentação binária com LaneNet no CARLA e aplicar controle PID sobre a direção do veículo. A modularidade permite evolução para controle lateral com MPC, aprendizado por reforço ou teste em ambientes reais.

Você pode usar este conhecimento como base para desenvolver veículos autônomos mais robustos e responsivos.

---
