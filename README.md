Read me file 
PyTorch implementation of Stable Diffusion from scratch

Download weights and tokenizer files:
Download vocab.json and merges.txt from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer and save them in the data folder
Download v1-5-pruned-emaonly.ckpt from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main and save it in the data folder

Tested fine-tuned models:
Just download the ckpt file from any fine-tuned SD (up to v1.5).

InkPunk Diffusion: https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main
Illustration Diffusion (Hollie Mengert): https://huggingface.co/ogkalu/Illustration-Diffusion/tree/main


***Why do we model data as distributions?***
• Imagine you’re a criminal, and you want to generate thousands of fake identities. Each fake identity, is made up of
variables, representing the characteristics of a person (Age, Height).
• You can ask the Statistics Department of the Government to give you statistics about the age and the height of the
population and then sample from these distributions.
Age: N(40, 302
) Height: N(120, 1002
)
• At first, you may sample from each distribution independently to create a
fake identity, but that would produce unreasonable pairs of (Age, Height).
• To generate fake identities that make sense, you need the joint
distribution, otherwise you may end up with an unreasonable pair of
(Age, Height)
• We can also evaluate probabilities on one of the two variables using
conditional probability and/or by marginalizing a variable.