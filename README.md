# 🎭 Dual LoRA Regional Control

Uma extensão para Stable Diffusion WebUI que permite aplicar diferentes LoRAs e prompts em regiões específicas da imagem.

## 🚀 Funcionalidades

- **Controle Regional**: Aplica diferentes LoRAs e prompts em regiões específicas da imagem
- **Múltiplos Modos de Divisão**: Vertical, horizontal ou máscara customizada
- **Integração com Prompt Principal**: Conecta automaticamente os prompts regionais com o prompt principal
- **Feathering**: Suavização das bordas entre regiões
- **Interface Intuitiva**: Interface gráfica fácil de usar

## 📦 Instalação

1. Copie a pasta `dual-lora` para o diretório `extensions` do seu Stable Diffusion WebUI
2. Reinicie o WebUI
3. A extensão aparecerá na aba "Scripts" da interface

## 🎯 Como Usar

### 1. Configuração Básica
- Ative a extensão marcando "✅ Enable Dual LoRA Regional Control"
- Configure o prompt principal no campo de prompt normal do WebUI

### 2. Configuração das Regiões

#### Região A (🅰️)
- **LoRA A**: Selecione o LoRA para a primeira região
- **Weight A**: Ajuste o peso do LoRA (0.0 - 2.0)
- **Prompt A**: Digite o prompt específico para esta região

#### Região B (🅱️)
- **LoRA B**: Selecione o LoRA para a segunda região
- **Weight B**: Ajuste o peso do LoRA (0.0 - 2.0)
- **Prompt B**: Digite o prompt específico para esta região

### 3. Configuração da Divisão

#### Modo de Divisão
- **Vertical**: Divide a imagem em esquerda/direita
- **Horizontal**: Divide a imagem em topo/baixo
- **Custom**: Use uma máscara personalizada

#### Split Ratio
- Controla a posição da divisão (0.1 - 0.9)
- 0.5 = divisão no centro

### 4. Ajustes Avançados

#### Feather Size
- Suaviza as bordas entre as regiões
- Valores maiores = bordas mais suaves

#### Blend Strength
- Controla a força da mistura entre as regiões

## 🔗 Conexão com Prompt Principal

A extensão automaticamente:
1. Captura o prompt principal do WebUI
2. Combina com os prompts regionais
3. Aplica os LoRAs selecionados
4. Mostra o prompt final combinado

### Exemplo de Uso

**Prompt Principal:**
```
beautiful landscape, high quality, detailed
```

**Prompt A:**
```
beautiful woman, portrait, detailed face
```

**Prompt B:**
```
cyberpunk city, neon lights, futuristic
```

**Resultado Final:**
```
beautiful landscape, high quality, detailed, (beautiful woman, portrait, detailed face:1.3), (cyberpunk city, neon lights, futuristic:1.3), <lora:woman_lora:1.0>, <lora:cyberpunk_lora:1.0>
```

## 🛠️ Troubleshooting

### Problemas Comuns

1. **LoRAs não aparecem na lista**
   - Verifique se os LoRAs estão na pasta `models/Lora`
   - Clique em "🔄 Refresh LoRAs"

2. **Prompts não se conectam**
   - Certifique-se de que a extensão está ativada
   - Verifique os logs no console do WebUI

3. **Erro de OpenCV**
   - A extensão tentará instalar automaticamente
   - Se falhar, instale manualmente: `pip install opencv-python`

## 📝 Logs

A extensão fornece logs detalhados no console:
- `📝 Prompt original`: Mostra o prompt original
- `📝 Prompt melhorado`: Mostra o prompt final combinado
- `🎯 Prompts regionais`: Mostra os prompts regionais
- `🔗 Conexão estabelecida`: Confirma que os prompts estão conectados

## 🤝 Contribuição

Sinta-se à vontade para contribuir com melhorias e correções!

## 📄 Licença

Este projeto está sob licença MIT. 
