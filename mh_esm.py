import torch
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

batch_converter = alphabet.get_batch_converter()
model.cuda()
model.eval() 

data = [
    ("protein1", "MRTDKEIFVSVDVETSGPIPGKYSMLSIGACVAFEPSKQFSCYLKPISEDFIPAAMEVTGLSLEKLHVDGLDPVDAMVQFKEWINSVVKEDETVVFVGFNASFDWSFINYYFHVYLGDNPFGIAALDIKSMYFGVSHASWRLTRSSEIAKVVKPETYGDHDALHDARYQAELFRLIDKLSEKKKLDR"),
    # ("protein2", "MRTDKEIFVSVDVETSGPIPGKYSMLSIGACVAFEPSKQFSCYLKPISEDFIPAAMEVTGLSLEKLHVDGLDPVDAMVQFKEWINSVVKEDETVVFVGFNASFDWSFINYYFHVYLGDNPFGIAALDIKSMYFGVSHASWRLTRSSEIAKVVKPETYGDHDALHDARYQAELFRLIDKLSEKKKLDR"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model.mh_sampling(batch_tokens.cuda())
