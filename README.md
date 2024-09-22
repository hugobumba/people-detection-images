# Detecção de Pedestres em Imagem

Este projeto utiliza o HOG (Histograms of Oriented Gradients) para detectar pedestres em uma imagem. O código é baseado na biblioteca OpenCV e utiliza técnicas de processamento de imagem.

## Requisitos

- Python 3.8.0
- OpenCV
- NumPy
- imutils

## Instalação

Para instalar as dependências, execute:

```bash
pip install opencv-python numpy imutils
```

## Uso

1. Coloque uma imagem no mesmo diretório do script.
2. Execute o script:

```bash
python pedestrian_detection.py
```

3. O programa abrirá uma janela mostrando a imagem com os pedestres detectados.

## Funcionamento

- O código inicializa o detector HOG.
- A imagem é lida e pré-processada.
- Os pedestres detectados são destacados com retângulos e etiquetas.
- O total de pedestres detectados é exibido na tela.