import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_TO_TERMINATORS = {
    "llama3": "<|eot_id|>",
    "deepseek": "<｜end▁of▁sentence｜>",
    "qwen": "<|im_end|>",
}


def get_model_name(model_dir: str):
    name = os.path.basename(model_dir).lower()
    if "llama3" in name:
        return "llama3"
    if "deepseek" in name:
        return "deepseek"
    if "qwen" in name:
        return "qwen"
    return None


def build_terminators(tokenizer, model_name):
    terminators = []

    if tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)

    if model_name in MODEL_TO_TERMINATORS:
        term_token = MODEL_TO_TERMINATORS[model_name]
        term_id = tokenizer.convert_tokens_to_ids(term_token)
        if term_id is not None and term_id != tokenizer.unk_token_id:
            terminators.append(term_id)

    # 去重
    terminators = list(dict.fromkeys(terminators))

    if not terminators:
        raise ValueError("No valid eos/terminator token ids found for this tokenizer.")

    return terminators


def predict(input_file, output_file, model, tokenizer, terminators):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    is_first = True

    for sample in tqdm(data, desc="Processing samples", unit="sample"):
        prompt = sample["input"]

        messages = [
            {
                "role": "system",
                "content": "请对用户输入的中文文本进行语法纠错，输出纠正后的句子，并给出简要纠错说明。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        # print("messages:", messages)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048,
        )

        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=512,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        new_tokens = generated_ids[0][model_inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        output_sample = {
            "input": prompt,
            "output": response
        }

        if is_first:
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump([output_sample], f_out, indent=4, ensure_ascii=False)
            is_first = False
        else:
            with open(output_file, "r", encoding="utf-8") as f_in:
                context = json.load(f_in)

            if not isinstance(context, list):
                raise ValueError("Error: The content of the JSON file is not a list.")

            context.append(output_sample)

            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump(context, f_out, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run base chat model for prediction.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name = get_model_name(args.model_dir)
    if model_name is None:
        raise ValueError(
            f"Unsupported model in {args.model_dir}. "
            "Expected model dir name containing one of: llama3, deepseek, qwen."
        )

    terminators = build_terminators(tokenizer, model_name)

    print("model_name:", model_name)
    print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("terminators:", terminators)

    predict(args.input_file, args.output_file, model, tokenizer, terminators)