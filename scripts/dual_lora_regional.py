import os
import torch
import gradio as gr
from modules import scripts, shared, sd_models, extra_networks, devices
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import on_cfg_denoiser
import numpy as np
from PIL import Image, ImageDraw
try:
    import cv2
except ImportError:
    print("OpenCV não encontrado. Instalando...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

class DualLoRARegional(scripts.Script):
    def __init__(self):
        super().__init__()
        self.active_regions = {}
        self.hooks_installed = False
        
    def title(self):
        return "Dual LoRA Regional Control"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("🎭 Dual LoRA Regional Control", open=False):
                enabled = gr.Checkbox(
                    label="✅ Enable Dual LoRA Regional Control", 
                    value=False,
                    elem_id="dual_lora_enabled"
                )
                
                with gr.Row():
                    with gr.Column(variant="panel"):
                        gr.HTML("<h4 style='color: #ff6b6b;'>🅰️ Region A</h4>")
                        lora_a = gr.Dropdown(
                            label="LoRA A",
                            choices=self.get_available_loras(),
                            value="None",
                            interactive=True,
                            elem_id="lora_a_dropdown"
                        )
                        weight_a = gr.Slider(
                            label="Weight A",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.05,
                            elem_id="weight_a_slider"
                        )
                        prompt_a = gr.Textbox(
                            label="Prompt A (específico para região A)",
                            placeholder="Ex: beautiful woman, portrait, detailed face",
                            lines=2,
                            elem_id="prompt_a_text"
                        )
                        
                    with gr.Column(variant="panel"):
                        gr.HTML("<h4 style='color: #4ecdc4;'>🅱️ Region B</h4>")
                        lora_b = gr.Dropdown(
                            label="LoRA B",
                            choices=self.get_available_loras(),
                            value="None",
                            interactive=True,
                            elem_id="lora_b_dropdown"
                        )
                        weight_b = gr.Slider(
                            label="Weight B",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.05,
                            elem_id="weight_b_slider"
                        )
                        prompt_b = gr.Textbox(
                            label="Prompt B (específico para região B)",
                            placeholder="Ex: cyberpunk city, neon lights, futuristic",
                            lines=2,
                            elem_id="prompt_b_text"
                        )
                
                with gr.Row():
                    split_mode = gr.Radio(
                        label="📐 Split Mode",
                        choices=[
                            ("Vertical (Esquerda/Direita)", "vertical"),
                            ("Horizontal (Topo/Baixo)", "horizontal"),
                            ("Máscara Customizada", "custom")
                        ],
                        value="vertical",
                        elem_id="split_mode_radio"
                    )
                    
                with gr.Row():
                    split_ratio = gr.Slider(
                        label="⚖️ Split Ratio (posição da divisão)",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        elem_id="split_ratio_slider"
                    )
                
                mask_upload = gr.Image(
                    label="📁 Custom Mask (Branco=Região A, Preto=Região B)",
                    type="pil",
                    visible=False,
                    elem_id="custom_mask_upload"
                )
                
                with gr.Row():
                    feather_size = gr.Slider(
                        label="🪶 Feather Size (suavização da borda)",
                        minimum=0,
                        maximum=100,
                        value=20,
                        step=2,
                        elem_id="feather_size_slider"
                    )
                    blend_strength = gr.Slider(
                        label="🌊 Blend Strength (força da mistura)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        elem_id="blend_strength_slider"
                    )
                
                with gr.Row():
                    refresh_loras = gr.Button("🔄 Refresh LoRAs", elem_id="refresh_loras_btn")
                    preview_mask = gr.Button("👁️ Preview Mask", elem_id="preview_mask_btn")
                
                mask_preview = gr.Image(
                    label="Mask Preview",
                    type="pil",
                    visible=False,
                    elem_id="mask_preview_img"
                )
                
                # Event handlers
                def update_mask_visibility(mode):
                    return gr.update(visible=(mode == "custom"))
                
                def update_lora_choices():
                    choices = self.get_available_loras()
                    return gr.update(choices=choices), gr.update(choices=choices)
                
                def generate_mask_preview(mode, ratio, custom_mask, width=512, height=512):
                    mask = self.create_region_mask(width, height, mode, ratio, custom_mask)
                    # Converte máscara para visualização
                    mask_vis = (mask * 255).astype(np.uint8)
                    mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
                    mask_colored[:, :, 0] = mask_vis  # Região A em vermelho
                    mask_colored[:, :, 2] = 255 - mask_vis  # Região B em azul
                    
                    mask_img = Image.fromarray(mask_colored)
                    return gr.update(value=mask_img, visible=True)
                
                split_mode.change(
                    fn=update_mask_visibility,
                    inputs=[split_mode],
                    outputs=[mask_upload]
                )
                
                refresh_loras.click(
                    fn=update_lora_choices,
                    outputs=[lora_a, lora_b]
                )
                
                preview_mask.click(
                    fn=generate_mask_preview,
                    inputs=[split_mode, split_ratio, mask_upload],
                    outputs=[mask_preview]
                )
        
        return [enabled, lora_a, weight_a, prompt_a, lora_b, weight_b, prompt_b, 
                split_mode, split_ratio, mask_upload, blend_strength, feather_size]
    
    def get_available_loras(self):
        """Obtém lista de LoRAs disponíveis no sistema"""
        loras = ["None"]
        
        # Procura em vários diretórios possíveis
        possible_dirs = [
            os.path.join(shared.models_path, "Lora"),
            os.path.join(shared.models_path, "lora"),
            os.path.join(shared.models_path, "LoRA"),
            "models/Lora",
            "models/lora"
        ]
        
        for lora_dir in possible_dirs:
            if os.path.exists(lora_dir):
                try:
                    for file in os.listdir(lora_dir):
                        if file.lower().endswith(('.safetensors', '.ckpt', '.pt', '.bin')):
                            if file not in loras:
                                loras.append(file)
                except Exception as e:
                    print(f"Erro ao listar LoRAs em {lora_dir}: {e}")
        
        return loras
    
    def create_region_mask(self, width, height, split_mode, split_ratio, custom_mask=None):
        """Cria máscara para dividir regiões"""
        if split_mode == "custom" and custom_mask is not None:
            # Processa máscara customizada
            mask = custom_mask.convert('L').resize((width, height), Image.Resampling.LANCZOS)
            mask_array = np.array(mask) / 255.0
        elif split_mode == "vertical":
            # Divisão vertical
            mask_array = np.zeros((height, width), dtype=np.float32)
            split_x = int(width * split_ratio)
            mask_array[:, :split_x] = 1.0
        else:  # horizontal
            # Divisão horizontal
            mask_array = np.zeros((height, width), dtype=np.float32)
            split_y = int(height * split_ratio)
            mask_array[:split_y, :] = 1.0
        
        return mask_array
    
    def apply_feather(self, mask, feather_size):
        """Aplica feathering suave nas bordas da máscara"""
        if feather_size <= 0:
            return mask
        
        # Converte para uint8 para processamento
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Aplica blur gaussiano para suavizar
        kernel_size = max(3, feather_size * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        try:
            blurred = cv2.GaussianBlur(
                mask_uint8, 
                (kernel_size, kernel_size), 
                feather_size / 3.0
            )
            return blurred.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Erro no feathering: {e}")
            return mask
    
    def process_prompts(self, p, prompt_a, prompt_b):
        """Processa e combina prompts regionais"""
        base_prompt = p.prompt if p.prompt else ""
        
        # Limpa prompts vazios
        prompt_a = prompt_a.strip() if prompt_a else ""
        prompt_b = prompt_b.strip() if prompt_b else ""
        
        # Combina prompts de forma inteligente
        if prompt_a and prompt_b:
            if base_prompt:
                p.prompt = f"{base_prompt}, ({prompt_a}:1.2), ({prompt_b}:1.2)"
            else:
                p.prompt = f"({prompt_a}:1.2), ({prompt_b}:1.2)"
        elif prompt_a:
            if base_prompt:
                p.prompt = f"{base_prompt}, ({prompt_a}:1.2)"
            else:
                p.prompt = f"({prompt_a}:1.2)"
        elif prompt_b:
            if base_prompt:
                p.prompt = f"{base_prompt}, ({prompt_b}:1.2)"
            else:
                p.prompt = f"({prompt_b}:1.2)"
    
    def process(self, p, enabled, lora_a, weight_a, prompt_a, lora_b, weight_b, 
                prompt_b, split_mode, split_ratio, custom_mask, blend_strength, feather_size):
        """Processa a geração com LoRAs regionais"""
        
        if not enabled:
            return
        
        print("🎭 Dual LoRA Regional: Iniciando processamento...")
        
        # Processa prompts
        self.process_prompts(p, prompt_a, prompt_b)
        
        # Cria máscaras regionais
        try:
            mask_a = self.create_region_mask(p.width, p.height, split_mode, split_ratio, custom_mask)
            mask_b = 1.0 - mask_a
            
            # Aplica feathering
            if feather_size > 0:
                mask_a = self.apply_feather(mask_a, feather_size)
                mask_b = self.apply_feather(mask_b, feather_size)
                # Renormaliza
                total_mask = mask_a + mask_b
                mask_a = mask_a / (total_mask + 1e-8)
                mask_b = mask_b / (total_mask + 1e-8)
            
            # Armazena configuração regional
            self.active_regions = {
                'enabled': True,
                'mask_a': mask_a,
                'mask_b': mask_b,
                'lora_a': lora_a if lora_a != "None" else None,
                'lora_b': lora_b if lora_b != "None" else None,
                'weight_a': weight_a,
                'weight_b': weight_b,
                'blend_strength': blend_strength,
                'width': p.width,
                'height': p.height
            }
            
            print(f"✅ Máscaras criadas: A={mask_a.sum():.0f}px, B={mask_b.sum():.0f}px")
            print(f"🎨 LoRAs: A={lora_a}({weight_a}), B={lora_b}({weight_b})")
            
        except Exception as e:
            print(f"❌ Erro no processamento: {e}")
            self.active_regions = {}
    
    def postprocess(self, p, processed, *args):
        """Limpeza após processamento"""
        if hasattr(self, 'active_regions'):
            self.active_regions = {}
        return processed