import tinker
<<<<<<< HEAD
from tinker.types import SampledSequence
import inspect
print(inspect.signature(SampledSequence))
=======
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker.types import SamplingParams

sc = tinker.ServiceClient()
tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
renderer = renderers.get_renderer("role_colon", tokenizer)

sampling_client = sc.create_sampling_client(
    model_path="tinker://5194866b-f63f-5c06-9398-c0f780df94fb:train:0/sampler_weights/multitask-8b-v5"
)
prompt = "Write a short sentence about the weather. Do not use any commas."
model_input = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
result = sampling_client.sample(
    num_samples=4,
    prompt=model_input,
    sampling_params=SamplingParams(
        max_tokens=200,
        temperature=1.0,
        stop=["User:", "\nUser", "user:"],
    ),
).result()

for i, s in enumerate(result.sequences):
    print(f"Sample {i+1}: {tokenizer.decode(s.tokens)[:150]}")
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
