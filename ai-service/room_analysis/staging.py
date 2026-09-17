import os
import re
import cv2
import numpy as np
from PIL import Image

AI_SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_staging_prompt(
    room_type="living_room",
    style="modern_luxury",
    user_description=None,
    detected_objects=None,
    free_space_pct=100.0,
):
    """
    Builds a professional, architectural interior design prompt for diffusion staging.
    Blends user description with room geometry preservation directives.
    """
    detected_names = [o.get("name", "") for o in (detected_objects or [])]

    # Base style adjectives
    style_keywords = {
        "modern_luxury": "luxury contemporary, refined elegance, high-end interior architecture, warm ambient lighting, 8k uhd architectural digest photography",
        "scandinavian": "nordic scandinavian design, light oak wood, minimalist clean aesthetic, cozy textures, warm neutral tones, soft daylight",
        "minimalist": "ultra-clean minimalist design, uncluttered spatial layout, sleek geometry, neutral palette, matte finishes",
        "industrial": "modern loft industrial aesthetic, dark metal accents, warm reclaimed wood, textured surfaces, studio lighting",
        "cozy_contemporary": "inviting contemporary living space, plush layered fabrics, warm atmospheric lighting, comfortable aesthetic",
    }
    style_str = style_keywords.get(style, style_keywords["modern_luxury"])

    # Base furniture ensembles per room type
    room_ensembles = {
        "living_room": (
            "A complete professionally staged luxury living room suite: "
            "a large contemporary sectional sofa with designer accent pillows, "
            "a textured woven area rug underneath the entire seating area, "
            "an elegant low-profile marble and walnut coffee table, "
            "a sleek minimalist TV media console against the side wall, "
            "a tall potted indoor fiddle-leaf fig plant in the corner, "
            "a slim brass standing floor lamp, "
            "and matching framed modern gallery canvas art centered on the blank back wall."
        ),
        "bedroom": (
            "A complete professionally staged luxury master bedroom suite: "
            "a king-size upholstered platform bed with layered textured linens and duvet, "
            "matching symmetrical nightstands with warm glowing bedside lamps, "
            "a large plush neutral area rug underneath the bed, "
            "a sleek minimalist wooden dresser console, "
            "and elegant framed abstract wall art above the headboard."
        ),
        "office": (
            "A complete professionally staged executive home office suite: "
            "a sleek solid oak executive desk, "
            "an ergonomic swivel leather desk chair, "
            "a modern open bookshelf with organized books and decor accents, "
            "a contemporary desk lamp, "
            "a low-pile geometric area rug, "
            "and tasteful framed architectural wall art."
        ),
        "gym": (
            "A complete professionally staged home fitness studio: "
            "clean interlocking rubber workout floor mats, "
            "a compact organized dumbbell rack with paired weights, "
            "an adjustable workout bench, "
            "a full-length frameless wall mirror reflecting natural light, "
            "and an exercise pull-up bar."
        ),
        "study": (
            "A tranquil modern study and reading nook: "
            "a spacious wooden study desk with laptop and lamp, "
            "comfortable ergonomic chair, "
            "floor-to-ceiling bookshelf filled with books, "
            "a plush armchair reading corner with floor lamp, "
            "and warm hardwood floor rug."
        ),
    }

    core_furniture = room_ensembles.get(room_type, room_ensembles["living_room"])

    # Preservation instruction
    preservation = (
        "Preserve the exact architectural structure of this empty room: "
        "keep the identical wall positions, window frames, door openings, ceiling lines, "
        "and flooring material. Seamlessly stage the furnishings inside this space."
    )

    # Blend custom user description if provided
    custom_clause = ""
    if user_description and user_description.strip():
        custom_clause = f" Custom styling request: {user_description.strip()}."

    full_prompt = f"{preservation} {core_furniture}{custom_clause} Style: {style_str}."

    negative_prompt = (
        "distorted walls, floating furniture, disconnected limbs, warped window frames, "
        "crooked perspective, low resolution, blurry, watermark, cartoonish, 2d sticker look, "
        "oversaturated, cluttered junk, broken geometry"
    )

    has_bed = any("bed" in (o.get("name", "")).lower() for o in (detected_objects or []))

    return {
        "positive": full_prompt,
        "negative": negative_prompt,
        "style": style,
        "room_type": room_type,
        "user_description": user_description or "",
        "has_bed": has_bed,
    }


def compress_user_prompt(description, room_type="living_room", style="modern_luxury"):
    """
    Intelligently extracts key interior design entities from long user descriptions
    and formats them into a dense, high-impact prompt strictly under 60 tokens for CLIP.
    """
    if not description or not description.strip():
        return None

    text = description.strip()

    style_keywords = {
        "modern_luxury": "luxury contemporary, Italian designer finishes, optimal spatial zoning, warm ambient lighting, 8k uhd architectural photography, sharp focus",
        "scandinavian": "nordic scandinavian design, light oak wood, minimalist clean aesthetic, cozy textures, warm neutral tones, 8k uhd architectural photography",
        "minimalist": "ultra-clean minimalist design, uncluttered spatial layout, sleek geometry, neutral palette, 8k uhd architectural digest",
        "industrial": "modern loft industrial aesthetic, dark metal accents, warm reclaimed wood, studio lighting, 8k uhd",
        "cozy_contemporary": "inviting contemporary living space, plush layered fabrics, warm atmospheric lighting, 8k uhd",
    }
    style_str = style_keywords.get(style, style_keywords["modern_luxury"])

    salient_patterns = [
        # Beds & Nursery
        r"(?:king|queen|double|master)[\w\s-]*bed",
        r"twin[\w\s-]*cribs?",
        r"crib[\w\s-]*setup",
        r"nursery[\w\s-]*setup",
        r"changing station",
        # Desks & Work
        r"study[\w\s-]*desk",
        r"work[\w\s-]*desk",
        r"executive[\w\s-]*desk",
        r"floating[\w\s-]*desk",
        r"ergonomic[\w\s-]*chair",
        r"office[\w\s-]*chair",
        r"gaming[\w\s-]*chair",
        r"monitor[\w\s-]*setup",
        r"playstation[\w\s-]*gaming",
        r"bookshelf",
        r"floating shelf",
        # Seating & Tables
        r"(?:curved\s+)?(?:boucl[ée]\s+)?sectional[\w\s-]*sofa",
        r"(?:2|3)[\w\s-]*seater sofa",
        r"armchair",
        r"couch",
        r"chaise lounge",
        r"(?:marble|fluted|walnut|oak)?\s*coffee table",
        r"dining[\w\s-]*table",
        r"dining[\w\s-]*chairs?",
        r"nightstands?",
        r"media console",
        r"tv console",
        r"wardrobe",
        r"dresser",
        r"credenza",
        # Greens & Decor
        r"(?:fiddle leaf fig|potted plant|indoor plant|plants?)",
        r"pothos vine",
        r"olive tree",
        r"wall art",
        r"canvas art",
        r"wood[\w\s-]*slat[\w\s-]*wall",
        # Textiles & Lighting
        r"(?:sheer|white)?\s*curtains",
        r"(?:area|textured|geometric|berber|wool)?\s*rugs?",
        r"(?:pendant|bedside|ambient|overhead|recessed|cove)?\s*lighting",
        r"lamps?",
        r"wall sconces?",
        # Fitness
        r"rubber[\w\s-]*mats?",
        r"dumbbell[\w\s-]*rack",
        r"workout bench",
        r"pull[\w\s-]*up bar",
        r"full-length mirror",
    ]

    extracted = []
    text_lower = text.lower()
    for pattern in salient_patterns:
        match = re.search(pattern, text_lower)
        if match:
            item_text = match.group(0).strip()
            item_text = re.sub(r"^(?:add a|position a|integrate a|place a|layer the|use)\s+", "", item_text)
            if item_text and item_text not in extracted:
                extracted.append(item_text)

    # Substring deduplication
    deduped = []
    for item in extracted:
        is_sub = False
        for i, d in enumerate(deduped):
            if item in d:
                is_sub = True
                break
            elif d in item:
                deduped[i] = item
                is_sub = True
                break
        if not is_sub:
            deduped.append(item)

    if deduped:
        top_items = ", ".join(deduped[:7])
        return (
            f"A professionally staged luxury {room_type.replace('_', ' ')}, {top_items}, "
            f"{style_str}, photorealistic 8k uhd, architectural digest, sharp focus"
        )

    clean_desc = re.sub(r"\s+", " ", text)[:160].strip()
    return f"A professionally staged {room_type.replace('_', ' ')}, {clean_desc}, {style_str}, photorealistic 8k, sharp focus"


def build_inpaint_prompt(prompt_dict):
    """
    Creates a token-efficient (< 65 tokens) diffusion inpainting prompt.
    Prioritizes user_description directly, and dynamically tailors furniture
    ensembles based on room_type, existing objects (e.g. bed present), and style.
    """
    desc = (prompt_dict.get("user_description") or "").strip()
    room_type = prompt_dict.get("room_type", "living_room")
    style = prompt_dict.get("style", "modern_luxury")
    has_bed = prompt_dict.get("has_bed", False)

    style_keywords = {
        "modern_luxury": "luxury contemporary, Italian designer finishes, optimal spatial zoning, warm ambient lighting, 8k uhd architectural photography, sharp focus",
        "scandinavian": "nordic scandinavian design, light oak wood, minimalist clean aesthetic, cozy textures, warm neutral tones, 8k uhd architectural photography",
        "minimalist": "ultra-clean minimalist design, uncluttered spatial layout, sleek geometry, neutral palette, 8k uhd architectural digest",
        "industrial": "modern loft industrial aesthetic, dark metal accents, warm reclaimed wood, studio lighting, 8k uhd",
        "cozy_contemporary": "inviting contemporary living space, plush layered fabrics, warm atmospheric lighting, 8k uhd",
    }
    style_str = style_keywords.get(style, style_keywords["modern_luxury"])

    # If user provided a specific custom description, compress and prioritize it!
    if desc:
        compressed = compress_user_prompt(desc, room_type=room_type, style=style)
        if compressed:
            inpaint_prompt = compressed
        else:
            inpaint_prompt = (
                f"A professionally staged {room_type.replace('_', ' ')}, {desc[:140]}, "
                f"tasteful complementary area rug and modern decor, {style_str}, photorealistic 8k, sharp focus"
            )
    else:
        # Defaults tailored per room type and context
        if room_type == "bedroom":
            if has_bed:
                inpaint_prompt = (
                    f"A modern master bedroom suite with dedicated study corner, sleek wooden study desk with chair, "
                    f"laptop and organizer, reading chair, modern area rug, warm lighting, {style_str}, photorealistic 8k"
                )
            else:
                inpaint_prompt = (
                    f"A luxury master bedroom suite, plush king-size upholstered bed with layered linens and pillows, "
                    f"matching nightstands with bedside lamps, large plush area rug, dresser, {style_str}, photorealistic 8k"
                )
        elif room_type == "office":
            inpaint_prompt = (
                f"A modern executive home office, sleek executive wood desk, ergonomic leather office chair, "
                f"open bookcase shelf with decor, geometric area rug, warm lamp, indoor plant, {style_str}, photorealistic 8k"
            )
        elif room_type == "gym":
            inpaint_prompt = (
                f"A modern home fitness studio, interlocking rubber workout floor mats, organized dumbbell rack with weights, "
                f"adjustable workout bench, full-length mirror, bright studio lighting, photorealistic 8k"
            )
        elif room_type == "study":
            inpaint_prompt = (
                f"A tranquil modern study and reading room, spacious study desk with laptop and lamp, ergonomic chair, "
                f"tall open bookshelf, cozy armchair, soft textured floor rug, {style_str}, photorealistic 8k"
            )
        else:  # living_room default
            inpaint_prompt = (
                f"A luxury living room suite, large comfortable sectional sofa with accent cushions, "
                f"elegant wooden coffee table, large textured area rug, sleek media console, indoor plant, {style_str}, photorealistic 8k"
            )

    neg_prompt = (
        "empty room, bare wooden floor, distorted furniture, floating objects, blurry, "
        "cartoonish, flat sticker, 2d cutouts, low quality, oversaturated, messy, cluttered junk, broken geometry, "
        "exterior, outdoor, landscape, mountains, sky, nature, distorted walls, crooked windows"
    )
    return inpaint_prompt, neg_prompt


class AIStagingEngine:
    """
    Engine to produce photorealistic staged room designs.
    Supports:
    - Cloud API (Fal.ai, Replicate, Gemini with billing)
    - Local GPU Diffusion Inpainting (RTX 3080 Ti)
    - Local GPU ControlNet Depth (if local model installed)
    - Coordinated Room Suite Compositing (high-fidelity local fallback)
    """

    def __init__(self):
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        self.fal_key = os.getenv("FAL_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def stage_room(
        self,
        original_image_path,
        depth_map_path,
        prompt_dict,
        output_path,
        floor_mask=None,
        window_mask=None,
        occupied_mask=None,
    ):
        """
        Attempts to stage the room using the best available staging provider.
        Returns dict with status and metadata.
        """
        prompt = prompt_dict["positive"]

        # 1. Try Cloud Staging API if credentials exist
        if self.replicate_token:
            res = self._stage_via_replicate(original_image_path, prompt, output_path)
            if res.get("success"):
                return res

        if self.fal_key:
            res = self._stage_via_fal(original_image_path, prompt, output_path)
            if res.get("success"):
                return res

        has_existing_furniture = (occupied_mask is not None and np.any(occupied_mask))

        # Mode A: If existing furniture exists, strictly preserve it via floor inpainting
        if has_existing_furniture:
            inpaint_res = self._stage_via_local_inpaint(
                original_image_path,
                floor_mask,
                prompt_dict,
                output_path,
                window_mask=window_mask,
                occupied_mask=occupied_mask,
            )
            if inpaint_res.get("success"):
                return inpaint_res

        # Mode B: Empty room / full conversion -> Use ControlNet Depth Img2Img for holistic high-end staging
        local_res = self._stage_via_local_controlnet(original_image_path, depth_map_path, prompt_dict, output_path)
        if local_res.get("success"):
            return local_res

        # Fallback to inpainting if controlnet was not used or failed
        if not has_existing_furniture:
            inpaint_res = self._stage_via_local_inpaint(
                original_image_path,
                floor_mask,
                prompt_dict,
                output_path,
                window_mask=window_mask,
                occupied_mask=occupied_mask,
            )
            if inpaint_res.get("success"):
                return inpaint_res

        # 4. Fallback to coordinated 2D ensemble
        return {
            "success": False,
            "provider": "local_ensemble",
            "message": "Using high-fidelity coordinated ensemble staging engine."
        }

    _inpaint_pipe = None

    @classmethod
    def get_inpaint_pipe(cls):
        if cls._inpaint_pipe is None:
            try:
                import torch
                if not torch.cuda.is_available():
                    return None
                from diffusers import AutoPipelineForInpainting
                print("[AIStagingEngine] Loading local inpainting pipeline on CUDA...")
                pipe = AutoPipelineForInpainting.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to("cuda")
                pipe.enable_attention_slicing()
                cls._inpaint_pipe = pipe
                print("[AIStagingEngine] Local inpainting pipeline ready on CUDA!")
            except Exception as e:
                print(f"[AIStagingEngine] Inpaint pipe loading error: {e}")
                return None
        return cls._inpaint_pipe

    def _stage_via_local_inpaint(
        self,
        image_path,
        floor_mask,
        prompt_dict,
        output_path,
        window_mask=None,
        occupied_mask=None,
    ):
        """
        Runs Diffusion Inpainting on the room's floor zone.
        Preserves original walls, windows, ceiling, and any pre-existing furniture,
        while generating photorealistic furnishings tailored to room_type and user_description.
        """
        try:
            import torch
            pipe = self.get_inpaint_pipe()
            if pipe is None:
                return {"success": False}

            from PIL import Image
            orig_bgr = cv2.imread(image_path)
            if orig_bgr is None:
                return {"success": False}
            height, width = orig_bgr.shape[:2]
            pil_orig = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))

            # If floor_mask is provided as boolean or uint8
            if floor_mask is not None:
                mask_data = (floor_mask > 0).astype(np.uint8) * 255
            else:
                # Fallback: lower 45% of image
                mask_data = np.zeros((height, width), dtype=np.uint8)
                mask_data[int(height * 0.55):, :] = 255

            # If there are existing objects (e.g. bed, existing table), strictly ensure they aren't masked
            if occupied_mask is not None and np.any(occupied_mask):
                occ_binary = (occupied_mask > 0).astype(np.uint8) * 255
                mask_data[occ_binary > 0] = 0

            # Dilate floor mask slightly upwards so sofa backrests or desk edges can stand naturally
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 45))
            mask_dilated = cv2.dilate(mask_data, kernel, iterations=1)

            # Strictly protect existing objects from inpainting dilation
            if occupied_mask is not None and np.any(occupied_mask):
                occ_dilated = cv2.dilate((occupied_mask > 0).astype(np.uint8) * 255, np.ones((11, 11), np.uint8))
                mask_dilated[occ_dilated > 0] = 0

            # Strictly protect window zones from inpainting
            if window_mask is not None:
                win_dilated = cv2.dilate((window_mask > 0).astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
                mask_dilated[win_dilated > 0] = 0

            mask_pil = Image.fromarray(mask_dilated).convert("L")

            # Resize to 768 on long edge
            scale = 768.0 / max(width, height)
            tw = int(width * scale // 8 * 8)
            th = int(height * scale // 8 * 8)
            input_img = pil_orig.resize((tw, th), Image.Resampling.LANCZOS)
            input_mask = mask_pil.resize((tw, th), Image.Resampling.LANCZOS)

            # Build tailored, token-efficient inpainting prompt
            inpaint_prompt, neg_prompt = build_inpaint_prompt(prompt_dict)

            print(f"[AIStagingEngine] Inpaint Prompt: {inpaint_prompt}")
            print("[AIStagingEngine] Running GPU Diffusion Inpainting on floor...")
            with torch.inference_mode():
                result_img = pipe(
                    prompt=inpaint_prompt,
                    negative_prompt=neg_prompt,
                    image=input_img,
                    mask_image=input_mask,
                    num_inference_steps=30,
                    guidance_scale=8.0,
                ).images[0]

            final_img = result_img.resize((width, height), Image.Resampling.LANCZOS)
            final_bgr = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
            # High-definition architectural detail enhancement
            gaussian = cv2.GaussianBlur(final_bgr, (0, 0), 2.0)
            sharpened = cv2.addWeighted(final_bgr, 1.15, gaussian, -0.15, 0)
            cv2.imwrite(output_path, sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[AIStagingEngine] HD Staged image saved to {output_path}")
            return {
                "success": True,
                "provider": "local_inpaint",
                "output_path": output_path,
                "inpaint_prompt": inpaint_prompt,
            }
        except Exception as e:
            print(f"[AIStagingEngine] Local inpainting notice: {e}")
            return {"success": False, "error": str(e)}

    _local_pipe = None

    @classmethod
    def get_local_pipe(cls):
        if cls._local_pipe is None:
            try:
                import torch
                if not torch.cuda.is_available():
                    return None
                from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
                print("[AIStagingEngine] Loading local ControlNet Depth Img2Img on RTX 3080...")
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11f1p_sd15_depth",
                    torch_dtype=torch.float16
                )
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to("cuda")
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.enable_attention_slicing()
                cls._local_pipe = pipe
                print("[AIStagingEngine] Local ControlNet Depth Img2Img pipeline ready on CUDA!")
            except Exception as e:
                print(f"[AIStagingEngine] Local pipe initialization notice: {e}")
                return None
        return cls._local_pipe

    def _stage_via_local_controlnet(self, image_path, depth_map_path, prompt_dict, output_path):
        """
        Invokes local GPU ControlNet Depth Img2Img pipeline.
        Conditions on the room's 3D depth geometry to preserve perspective and window/wall structure,
        while holistically styling the space with curtains, lighting, and photorealistic furniture.
        """
        try:
            import torch
            pipe = self.get_local_pipe()
            if pipe is None:
                return {"success": False}

            from PIL import Image

            orig_bgr = cv2.imread(image_path)
            if orig_bgr is None:
                return {"success": False}
            height, width = orig_bgr.shape[:2]
            pil_orig = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))

            # Resize to 768 on long edge maintaining aspect ratio divisible by 8
            scale = 768.0 / max(width, height)
            target_w = int(width * scale // 8 * 8)
            target_h = int(height * scale // 8 * 8)
            input_img = pil_orig.resize((target_w, target_h), Image.Resampling.LANCZOS)

            # Load depth map or compute from image
            if depth_map_path and os.path.exists(depth_map_path):
                depth_cv = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
            else:
                from room_analysis.depth import estimate_depth
                depth_cv = estimate_depth(image_path)

            if depth_cv is None:
                return {"success": False}

            depth_pil = Image.fromarray(depth_cv).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)

            # Build condensed, high-impact prompt
            desc = prompt_dict.get("user_description", "")
            room_type = prompt_dict.get("room_type", "living_room")
            style = prompt_dict.get("style", "modern_luxury")

            condensed = compress_user_prompt(desc, room_type=room_type, style=style)
            if not condensed:
                condensed, _ = build_inpaint_prompt(prompt_dict)

            neg_prompt = (
                "blurry, low quality, commercial office, drop ceiling tiles, empty room, "
                "distorted furniture, deformed, oversaturated, messy, watermark, flat cartoon, "
                "exterior, outdoor landscape, mountains, sky, deformed windows, distorted walls, crooked perspective"
            )

            print(f"[AIStagingEngine] ControlNet Prompt: {condensed}")
            print(f"[AIStagingEngine] Generating photorealistic staged design on GPU...")
            with torch.inference_mode():
                result_img = pipe(
                    prompt=condensed,
                    negative_prompt=neg_prompt,
                    image=input_img,
                    control_image=depth_pil,
                    strength=0.82,
                    controlnet_conditioning_scale=0.80,
                    num_inference_steps=30,
                    guidance_scale=8.0,
                ).images[0]

            final_img = result_img.resize((width, height), Image.Resampling.LANCZOS)
            final_bgr = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
            # High-definition architectural detail enhancement
            gaussian = cv2.GaussianBlur(final_bgr, (0, 0), 2.0)
            sharpened = cv2.addWeighted(final_bgr, 1.15, gaussian, -0.15, 0)
            cv2.imwrite(output_path, sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[AIStagingEngine] Photorealistic HD staged image saved to {output_path}")
            return {
                "success": True,
                "provider": "local_controlnet_img2img",
                "output_path": output_path,
                "staged_prompt": condensed,
            }
        except Exception as e:
            print(f"[AIStagingEngine] Local ControlNet staging notice: {e}")
            return {"success": False, "error": str(e)}

    def _stage_via_replicate(self, image_path, prompt, output_path):
        """Invoke Replicate Virtual Staging / ControlNet model."""
        try:
            import replicate
            client = replicate.Client(api_token=self.replicate_token)
            with open(image_path, "rb") as img_file:
                output = client.run(
                    "jagilley/controlnet-interior-design:850518650a43b5e402b8f72382f8779252f87071b7648492d7b565000d0ee9d3",
                    input={
                        "image": img_file,
                        "prompt": prompt,
                        "num_samples": "1",
                        "image_resolution": "768",
                    }
                )
            if output and len(output) > 0:
                import urllib.request
                urllib.request.urlretrieve(str(output[0]), output_path)
                return {
                    "success": True,
                    "provider": "replicate",
                    "output_path": output_path,
                }
        except Exception as e:
            print(f"[AIStagingEngine] Replicate staging notice: {e}")
        return {"success": False}

    def _stage_via_fal(self, image_path, prompt, output_path):
        """Invoke Fal.ai staging model."""
        try:
            import fal_client
            res = fal_client.subscribe(
                "fal-ai/controlnet-depth",
                arguments={
                    "image_url": image_path,
                    "prompt": prompt,
                }
            )
            if res and "images" in res and len(res["images"]) > 0:
                import urllib.request
                urllib.request.urlretrieve(res["images"][0]["url"], output_path)
                return {
                    "success": True,
                    "provider": "fal",
                    "output_path": output_path,
                }
        except Exception as e:
            print(f"[AIStagingEngine] Fal staging notice: {e}")
        return {"success": False}
