text='''
Farmers were working hard in the fields, preparing the soil and planting seeds for
the next season. The sun was bright, and the air smelled of earth and fresh grass.
The Indian Premier League (IPL) is the biggest cricket league in the world. People
all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates
fear in cities and villages. When such attacks happen, they leave behind pain and
sadness. To fight terrorism, we need strong laws, alert security forces, and support
from people who care about peace and safety.
'''
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_experimental.text_splitter import SemanticChunker



embedding= OllamaEmbeddings(
    model="mxbai-embed-large",
)
hf_embedding = HuggingFaceEmbeddings(
    model_name=r"E:\LangChain\models\bge-large-en",
)

splitter = SemanticChunker(
    embeddings=embedding,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

hf_splitter = SemanticChunker(
    embeddings=hf_embedding,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

result = splitter.create_documents([text])
hf_result = hf_splitter.create_documents([text])

for i, doc in enumerate(hf_result, 1):
    print(f"\n--- Semantic Chunk {i} ---\n{doc.page_content}")


'''
Breakpoints:
This chunker works by determining when to "break" apart sentences. This is done by looking for differences in embeddings between any two sentences. When that difference is past some threshold, then they are split.
There are a few ways to determine what that threshold is, which are controlled by the breakpoint_threshold_type kwarg.
Note: if the resulting chunk sizes are too small/big, the additional kwargs breakpoint_threshold_amount and min_chunk_size can be used for adjustments

Percentile
The default way to split is based on percentile. In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split. The default value for X is 95.0 and can be adjusted by the keyword argument breakpoint_threshold_amount which expects a number between 0.0 and 100.0.

Standard Deviation
In this method, any difference greater than X standard deviations is split. The default value for X is 3.0 and can be adjusted by the keyword argument breakpoint_threshold_amount.

Interquartile
In this method, the interquartile distance is used to split chunks. The interquartile range can be scaled by the keyword argument breakpoint_threshold_amount, the default value is 1.5.

Gradient
In this method, the gradient of distance is used to split chunks along with the percentile method. This method is useful when chunks are highly correlated with each other or specific to a domain e.g. legal or medical. The idea is to apply anomaly detection on gradient array so that the distribution become wider and easy to identify boundaries in highly semantic data. Similar to the percentile method, the split can be adjusted by the keyword argument breakpoint_threshold_amount which expects a number between 0.0 and 100.0, the default value is 95.0.

'''