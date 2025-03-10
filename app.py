import os
import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pydantic import BaseModel
from typing import Optional

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

app = FastAPI(
    title="Qwen2.5-VL Text Extraction API",
    description="API for extracting text from images using Qwen2.5-VL model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
processor = None

def resize_if_needed(image, max_size=1024):
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

@app.on_event("startup")
async def startup_event():
    global model, processor

    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: GPU not available, using CPU instead. This will be very slow.")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        offload_folder="offload",
    )

    model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("Model and processor loaded successfully")

class TextExtractionResponse(BaseModel):
    extracted_text: str

@app.post("/extract-text", response_model=TextExtractionResponse)
async def extract_text(
    file: UploadFile = File(...),
    user_prompt: Optional[str] = Form(None)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if user_prompt is None:
        user_prompt = (
            "Please transcribe all visible text from this image exactly as it appears. "
            "Preserve the layout, formatting, and structure of tables, lists, and paragraphs. "
            "Include all visible text, whether printed or handwritten, "
            "but strictly exclude any text that has been scratched out, crossed out, or deliberately marked for deletion. "
            "Your focus is to provide a clean and accurate transcription of only the text that was meant to be read."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        image = resize_if_needed(image)

        extracted_text = process_image(image, user_prompt)
        return {"extracted_text": extracted_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

def process_image(image, user_prompt):
    global model, processor

    if model is None or processor is None:
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: GPU not available, using CPU instead. This will be very slow.")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            offload_folder="offload",
        )
        model.gradient_checkpointing_enable()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print("Model and processor loaded successfully")

    system_prompt = (
        "You are an expert in extracting text from images. "
        "Your task is to transcribe all visible text from the provided image exactly as it appears, "
        "while strictly excluding any text that has been scratched out, crossed out, or deliberately marked for deletion. "
        "Your goal is to provide a clean and accurate transcription of only the text that was intended to be read."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    torch.cuda.empty_cache()
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    torch.cuda.empty_cache()

    return output_text[0]

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)