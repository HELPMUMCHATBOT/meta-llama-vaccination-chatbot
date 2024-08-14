---
library_name: transformers
tags:
- vaccination
- immunization
- chatbot
- healthcare
language:
- en
---

# Model Card for HelpMum Vax-Llama-1

The HelpMum Vax-Llama-1 is an advanced language model designed to provide accurate and relevant information about vaccinations and immunizations. It is fine-tuned from the Llama 3.1 8B model and built using the Hugging Face Transformers framework. This model has 8 billion parameters and is optimized for delivering precise responses to queries related to vaccination safety, schedules, and more.

## Model Details

### Model Description

The HelpMum Vax-Llama-1 model is a specialized chatbot model developed to enhance the dissemination of vaccination-related information. It has been fine-tuned from the Llama 3.1 8B base model, using a diverse dataset of vaccination queries and responses. This model aims to provide reliable information to users, helping them make informed decisions about vaccinations.

- **Developed by:** HelpMum
- **Funded by:** HelpMum
- **Shared by:** HelpMum
- **Model type:** Transformer-based language model
- **Language(s) (NLP):** English
- **Finetuned from model:** Llama 3.1 8B

### Model Sources

- **Repository:** [HelpMumHQ/vax-llama-1](https://huggingface.co/HelpMumHQ/vax-llama-1)

## Uses

### Direct Use

The model can be directly used to answer queries related to vaccinations and immunizations without any further fine-tuning. It is suitable for integration into chatbots and other automated response systems in healthcare settings.

### Downstream Use

The model can be fine-tuned for specific tasks or integrated into larger ecosystems and applications that require accurate vaccination information dissemination.

### Out-of-Scope Use

The model is not intended for use in generating medical advice beyond vaccination information. It should not be used for diagnosing medical conditions or providing treatment recommendations.

## Bias, Risks, and Limitations

The model is trained on a dataset of vaccination-related information, which may not cover all possible queries or scenarios. Users should be aware of potential biases in the data and limitations in the model's knowledge. It is essential to consult healthcare professionals for personalized medical advice.

### Recommendations

Users should ensure that the model is used in contexts where it can provide valuable information while being aware of its limitations. For critical medical decisions, consultation with healthcare professionals is recommended.

## How to Get Started with the Model

Use the following code to get started with the Vax-Llama-1 model:

```python
!pip install -q -U transformers
!pip install -q -U bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('HelpMumHQ/vax-llama-1')
model = AutoModelForCausalLM.from_pretrained('HelpMumHQ/vax-llama-1')

def generate_response(user_message):
    tokenizer.chat_template = "{%- for message in messages %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' if message['role'] == 'user' else ' ' + message['content'] + ' ' + eos_token }}{%- endfor %}"
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = (text[text.find('[/INST]') + len('[/INST]'):text.find('[INST]', text.find('[/INST]') + len('[/INST]'))] if text.find('[INST]', text.find('[/INST]') + len('[/INST]')) != -1 else text[text.find('[/INST]') + len('[/INST]'):]).strip().split('[/INST]')[0].strip()
    return response

# Sample usage
user_message = "Are vaccines safe for pregnant women?"
response = generate_response(user_message)
print(response)
```

## Training Details

### Training Data

The data for this model was collected from HelpMum's extensive database of vaccination-related queries and responses, which includes real-world interactions and expert-verified information.

### Testing Data, Factors & Metrics

#### Testing Data

The testing data was a separate subset of vaccination-related queries to evaluate the model's performance accurately.

#### Factors

The evaluation considered various factors, including the accuracy and relevance of responses, latency, and token allowance.

#### Metrics

- **Loss:** 0.3554
- **Runtime:** 195.8647 seconds
- **Samples per Second:** 0.735

### Results

The Vax-Llama-1 model performed well in delivering accurate and relevant responses to vaccination queries, with high user satisfaction and efficiency.

#### Summary

The model demonstrated robust performance across various evaluation metrics, making it a reliable tool for vaccination information dissemination.

## Model Examination

The model underwent rigorous testing and evaluation to ensure it meets the desired performance standards for accuracy and relevance.

## Technical Specifications

### Model Architecture and Objective

The Vax-Llama-1 is a transformer-based language model built on the Llama 3.1 architecture, designed to generate accurate responses to vaccination-related queries.

### Compute Infrastructure

#### Software

- **Framework:** Transformers (Hugging Face)
- **Programming Language:** Python

## Citation

**BibTeX:**

```bibtex
@misc {helpmumhq_2024,
    author       = { {HelpMumHQ} },
    title        = { vax-llama-1 (Revision 033a456) },
    year         = 2024,
    url          = { https://huggingface.co/HelpMumHQ/vax-llama-1 },
    doi          = { 10.57967/hf/2793 },
    publisher    = { Hugging Face }
}
```

## Glossary

- **Transformer:** A type of neural network architecture used for natural language processing tasks.
- **Fine-Tuning:** The process of taking a pre-trained model and further training it on a specific task or dataset.
- **Tokenization:** The process of converting text into a format that can be used by the model, typically involving splitting text into tokens.

## More Information

For more details and access to the model, visit [HelpMumHQ/vax-llama-1](https://huggingface.co/HelpMumHQ/vax-llama-1).

## Model Card Authors

HelpMum Tech Team

## Model Card Contact

For questions or feedback, please contact [HelpMum](mailto:tech@helpmum.org).
