# Documentação Detalhada do Código de Processamento de Vídeo para Detecção de Faixas

## Visão Geral
Este código implementa um sistema de processamento de vídeo para detecção de faixas em uma estrada utilizando a biblioteca OpenCV e MoviePy. Ele realiza um conjunto de operações de processamento de imagem, incluindo conversão para escala de cinza, detecção de bordas, seleção de região de interesse e aplicação da Transformada de Hough para identificar e desenhar linhas representando as faixas da pista.

## Dependências
O código utiliza as seguintes bibliotecas:

- **numpy**: Para manipulação de arrays numéricos.
- **pandas**: Não é utilizado explicitamente no código e pode ser removido.
- **cv2 (OpenCV)**: Para manipulação e processamento de imagens.
- **moviepy**: Para manipulação e edição de vídeos.
- **moviepy.video.io.VideoFileClip**: Para carregar e processar vídeos frame a frame.

Para instalar as dependências necessárias, utilize:
```bash
pip install numpy opencv-python moviepy
```

## Estrutura e Conceitos Abordados
O código está estruturado em várias funções modulares para realizar diferentes tarefas do processamento de imagem:

1. **Seleção de região de interesse (`region_selection`)**: Cria uma máscara para destacar a região onde as faixas da estrada estão normalmente localizadas.
2. **Transformada de Hough (`hough_transform`)**: Detecta linhas em uma imagem de bordas usando a Transformada de Hough Probabilística.
3. **Cálculo de inclinação e intercepto (`average_slope_intercept`)**: Determina as características das linhas da faixa.
4. **Conversão de equação de reta para coordenadas de pixel (`pixel_points`)**: Converte as linhas detectadas para um formato utilizável no desenho.
5. **Desenho das faixas na imagem (`draw_lane_lines`)**: Renderiza as linhas detectadas na imagem original.
6. **Processamento de um frame (`frame_processor`)**: Aplica todas as etapas de processamento em uma imagem.
7. **Processamento de vídeo (`process_video`)**: Aplica o processamento em cada frame de um vídeo e salva o vídeo processado.

## Explicação das Funções
### `region_selection(image)`
Cria uma máscara para selecionar a região da estrada, com um polígono delimitado pelas coordenadas:
- **Canto inferior esquerdo**: 10% da largura e 95% da altura.
- **Topo esquerdo**: 40% da largura e 60% da altura.
- **Canto inferior direito**: 90% da largura e 95% da altura.
- **Topo direito**: 60% da largura e 60% da altura.

Isso garante que apenas a região onde as faixas normalmente estão será processada.

### `hough_transform(image)`
Aplica a **Transformada de Hough** para detectar linhas na imagem de bordas:
- `rho = 1`: Unidade de resolução para distância radial.
- `theta = np.pi / 180`: Resolução angular em radianos.
- `threshold = 20`: Define o número mínimo de interseções para considerar uma linha.
- `minLineLength = 10`: Define o comprimento mínimo de uma linha detectada.
- `maxLineGap = 250`: Define a máxima separação entre segmentos para ser considerada uma linha contínua.

### `average_slope_intercept(lines)`
Agrupa as linhas detectadas em faixas esquerda e direita:
- Calcula a inclinação (`slope`) e o intercepto (`intercept`).
- Filtra as linhas em faixas esquerda (inclinação negativa) e direita (inclinação positiva).
- Retorna a média ponderada das linhas detectadas para suavizar a detecção.

### `pixel_points(y1, y2, line)`
Converte a equação da reta detectada para coordenadas de pixel:
- Calcula os pontos `(x1, y1)` e `(x2, y2)` com base na equação da reta `y = slope * x + intercept`.

### `lane_lines(image, lines)`
Determina as linhas finais das faixas:
- Define `y1` como a altura total da imagem (base da imagem).
- Define `y2` como 60% da altura (onde as linhas devem terminar).
- Obtém as coordenadas de pixel para as faixas esquerda e direita.

### `draw_lane_lines(image, lines, color=[139, 0, 0], thickness=12)`
Desenha as linhas das faixas na imagem:
- Cria uma imagem preta e desenha as linhas detectadas nela.
- Sobrepõe a imagem processada à imagem original usando `cv2.addWeighted()`.

### `frame_processor(image)`
Executa todo o pipeline de processamento em uma imagem:
1. Converte para escala de cinza.
2. Aplica filtro Gaussiano para suavização.
3. Usa o detector de bordas Canny.
4. Aplica a seleção de região de interesse.
5. Aplica a Transformada de Hough para detectar linhas.
6. Desenha as faixas na imagem original.
7. Retorna a imagem processada.

### `process_video(test_video, output_video)`
Processa um vídeo inteiro:
1. Carrega o vídeo de entrada usando `VideoFileClip()`.
2. Aplica `frame_processor()` a cada frame do vídeo.
3. Salva o vídeo processado.

## Execução
Para executar o código:
```python
python3 lane_project.py
```
Isso processará o vídeo `input.mp4` e salvará o resultado em `output.mp4`.

## Melhorias Possíveis
- Ajuste dinâmico dos parâmetros da Transformada de Hough para diferentes condições de iluminação.
- Melhor filtragem de ruído para evitar a detecção de linhas espúrias.
- Implementação de um algoritmo de tracking para estabilização das linhas detectadas.

Este código fornece um pipeline funcional para detecção de faixas em vídeos de estrada e pode ser aprimorado para casos mais complexos.


