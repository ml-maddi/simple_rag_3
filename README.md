# simple_rag_3

## create a virtual environment

`python -m venv venv`

### for windows activate the virtual env using

`.\venv\Scripts\activate`

### for mac/linux

`source venv/bin/activate`

## install required python packages

`pip install -r requirements.txt`

## install tessaract for text extraction

### on linux/mac

`sudo apt install tesseract-ocr tesseract-ocr-ben`
`sudo apt update`
`sudo apt install poppler-utils`

### on windows

`download from https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe`

`while installing under 'Additional language data' select 'Bengali'`
`add tesseract to path `
`default path: C:\Program Files\Tesseract-OCR\`

`2. Add the Path to Windows Environment Variables
Press Win + S and search for "Environment Variables"

Click “Edit the system environment variables”

In the System Properties window, click the “Environment Variables…” button

Under “System variables”, scroll to find Path and click Edit

Click New and add:

C:\Program Files\Tesseract-OCR\
Click OK on all dialogs to save

🔹 3. Restart Your Terminal or VSCode`

## collect gemini api key

`go to this url https://aistudio.google.com/prompts/new_chat and create an api key and place it in ".env" file `

## run a web app

`uvicorn main:app --reload`

## test out the web app online here

`https://really-uncommon-rat.ngrok-free.app/`

# api calling locally

### api status

`curl http://127.0.0.1:8000/api/status`

## sample querying

### on linux/mac

`curl -X POST http://127.0.0.1:8000/api/chat \
-H "Content-Type: application/json" \
-d '{"query": "who rejects the marriage"}'`

### on windows (powershell)

`curl -Uri http://127.0.0.1:8000/api/chat -Method POST -ContentType "application/json" -Body '{"query": "অপরিচিতা গল্পে অনুপমের মামার সঙ্গে কার তুলনা করা হয়েছে?"}'`

# Used tools

llm = gemini-2.5-flash<br>
embedding_model = Qwen/Qwen3-Embedding-0.6B<br>
vectorstore = faiss-cpu<br>

# Used libraries

fastapi[all]<br>
nest-asyncio<br>
pyngrok<br>
uvicorn<br>
python-socketio[asyncio]<br>
langchain<br>
langchain_community<br>
google-generativeai<br>
langchain-google-genai<br>
faiss-cpu<br>
pymupdf<br>
python-dotenv<br>
langchain_huggingface<br>
pytesseract <br>
pdf2image<br>
sentence-transformers<br>

# sample queries,outputs

Question: who is kalyani's brother <br>
Answer: Sorry, I am unable to answer this question.<br>

Question: who rejects the marriage<br>
Answer: Shambhunath Sen, Kalyani's father, rejects the marriage.<br>

Question: who is kalyani<br>
Answer: Kalyani is a character in the story. Her father's name is Shambhunath Sen, who is a doctor in Kanpur. At the time of her proposed marriage, Kalyani was 16 or 17 years old. Her marriage was called off due to the groom's family's greed and lack of self-respect. After this incident, Kalyani took a vow to educate girls. She is portrayed as a strong, self-respecting woman.<br>

Question: who is kalyani's father<br>
Answer: Kalyani's father is Shambhunath Sen.<br>

Question: অনুপমের পুরো নাম কি<br>
Answer: দুঃখিত, আপনার প্রশ্নটির উত্তর আমার জানা নেই।<br>

Question: গল্পের সারাংশ বোলো<br>
Answer: "অপরাজিতা" গল্পটি অনুপম নামের এক যুবককে কেন্দ্র করে আবর্তিত হয়েছে, যে তার মামার উপর সম্পূর্ণ নির্ভরশীল এবং ব্যক্তিত্বহীন। তার বাবা উকিল হিসেবে প্রচুর অর্থ উপার্জন করলেও ভোগ করার সুযোগ পাননি। অনুপমের বয়স সাতাশ হলেও সে মায়ের কোলে গণেশের ছোট ভাইয়ের মতো। তার মামা, যিনি তার ভাগ্যের প্রধান এজেন্ট, ধনী পাত্রী পছন্দ করেন না, কিন্তু যৌতুকের প্রতি তার তীব্র আসক্তি ছিল।

মামার পছন্দ অনুযায়ী, কানপুরের ডাক্তার শম্ভুনাথ সেনের পনেরো বছর বয়সী মেয়ে কল্যাণীর সাথে অনুপমের বিয়ের কথা হয়। বিয়ের দিন মামা গহনা যাচাই করার জন্য একজন জহুরিকে সাথে নিয়ে আসেন। এই ঘটনায় কল্যাণীর বাবা শম্ভুনাথ সেন অত্যন্ত অপমানিত বোধ করেন এবং বিয়ে ভেঙে দেন। তিনি বলেন, "ঠাট্টা তো আপনিই করিয়া সারিয়াছেন।"

এই ঘটনায় অনুপম লজ্জিত ও অপমানিত হয়, কিন্তু মামার বিরুদ্ধে কোনো প্রতিবাদ করতে পারে না। পরবর্তীতে, অনুপম তীর্থযাত্রায় বেরিয়ে ট্রেনে কল্যাণীর সাথে আবার দেখা করে। কল্যাণী তখন মেয়েদের পড়ানোর কাজে নিজেকে নিয়োজিত করেছে এবং দেশের সেবায় ব্রতী হয়েছে। সে আর বিয়ে করেনি। অনুপম কল্যাণীর আত্মমর্যাদা ও স্বাধীনচেতা মনোভাব দেখে মুগ্ধ হয় এবং তার প্রতি এক গভীর আকর্ষণ অনুভব করে। গল্পটি যৌতুক প্রথার কুফল এবং নারীর আত্মমর্যাদা ও স্বাধীন ব্যক্তিত্বের জয়কে তুলে ধরেছে। অনুপম তার নিষ্ক্রিয় জীবনের ব্যর্থতা উপলব্ধি করে এবং কল্যাণীর সান্নিধ্য পাওয়ার আকাঙ্ক্ষা নিয়েই তার জীবন কাটে।<br>

Question: what is the capital of Bangladesh<br>
Answer: Sorry, I am unable to answer this question.<br>

Question: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?<br>
Answer: কল্যাণীর বিয়ের সময় তার প্রকৃত বয়স ছিল ১৫ বছর।<br>

Question: who is kalyani's father<br>
Answer: Kalyani's father is Shambhunath Sen.<br>

Question: who are the main characters of the story<br>
Answer: The main characters of the story 'Aparichita' are Anupam (the narrator), Kalyani, Anupam's uncle, and Shambhunath Sen (Kalyani's father).<br>

Question: do kalyani and onupom finally get married<br>
Answer: No, Kalyani and Anupam do not get married in the story. The marriage is called off by Kalyani's father due to the groom's uncle's behavior regarding the dowry. Kalyani herself later states that she will not marry.<br>

Question: is it a sad story, or what is the story type of it... what could be tagline of this story<br>
Answer: The story about Kalyani and Anupam, titled "Aparichita," is a prose story, specifically a confessional narrative told from the perspective of the protagonist, Anupam.

Yes, it is considered a sad story, particularly for Anupam, who lives with regret and mental anguish due to his inability to act and the broken marriage proposal.

A suitable tagline for the story could be: "A tale of unfulfilled love and the anguish of a lost opportunity."<br>

# evaluation metric

as the evaluation metric i have used manual fact checking and also llm as a judge method. First i provided the pdf to strong llm model like "chatgpt" to help me find top 50 facts about the pdf or story... i then asked the llm to provide questions related to the facts , where the fact will give the direct answers.... i asked these questions to system and check manually about its performance... then as a metric of complex accuracy evaluation, i again asked the LLM to generate another 100 multi hop complex questions regarding the previously found top 50 facts from the story.... and also asked it to generate the correct answers for those questions according to it..... finally i asked these same multi hop questions to our system, and stored the generated answers , alongwith the retrieved chunked documents which were used to generated the answer from my system.... finally i used the large LLM "chatgpt" again to rate our system's answer Groundedness quality and to rate the context documents in terms of "Relevance"

# answers to questions

Question: What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?
Answer:
Initially, I used PyMuPDF library to extract the textual contents from the pdf file. The reason behind choosing this library is that, its one of the best performing library in extracting content from complex pdf files without needing to render the page, making it significantly faster and less resource-intensive than OCR.It's the ideal choice when a PDF has a clean, standard text layer. Though the library was able to extract textual content, but the texts were appearing as gibberish content, overlapping over one another, suggesting there may be issue with Unicode character misalignment or font encoding errors. As a result extracted contents were appearing like following:
“"৫৫। মামািব্ার্হকিিোত্রাপকথিসীমানাকতদূি? (ক) আন্দামান পেিন্ত (খ) যকান্নগি পেিন্ত (গ) কানপুি পেিন্ত (ঘ) হাও়ো পেিন্ত ৫৬। র্ব্ব্াকহিকতর্দনপূকব্িঅনুপকমিসাকথতািশ্বশুকিিসাক্ষাৎহ ? (ক) ২ র্দন (খ) ৩ র্দন (গ) ৪ র্দন (ঘ) ৫ র্দন ৫৭। 'র্তর্নব়্েইচুপচাপ' এখাকনকািকথাব্লাহক কি? (ক) মামা (খ) হর্িি (গ) িম্ভুনাথ (ঘ) মা ৫৮। 'র্তর্নর্কিুকতইঠককব্ননা' কািপ্রসকঙ্গব্লাহক কি? (ক) মামা (খ) মা (গ) র্ব্নুদাদা (ঘ) হর্িি ৫৯। 'অপর্ির্চতা' গকেযকানসম অনুপমর্ব্নুদাদািব্ার়্েকতযেত? (ক) সন্ধা (খ) িা”

To overcome this issue, I then opted for Optical Character Recognition (OCR) approach with Tesseract. While this method is more computationally expensive and slower than direct extraction (as it must first render the page as an image), it successfully bypassed the encoding issues by 'reading' the characters visually, ensuring the Bangla text was captured accurately.

Question: What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?
Answer For my chunking strategy, I used “RecursiveCharacterTextSplitter” from langchain.
I chose this method because it's highly effective at keeping semantically related content together without having any semantic awareness itself. It works by attempting to split text hierarchically along natural boundaries, starting with double newlines (paragraphs), then single newlines (sentences), and finally spaces.
This structural approach is beneficial for semantic retrieval because it tends to create chunks that represent complete thoughts or paragraphs. These coherent chunks can then be more meaningfully converted into vector embeddings by the language model. To implement this, I chose a
chunk size of 300 characters and a chunk overlap of 50 characters. The overlap ensures that context is not lost for sentences that fall on the boundary between two chunks, which is vital for accurate retrieval. The smaller chunk size is used so that not too much diverse content is being put within a single chunk, also making semantic searching easier.

Question: What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?
Answer: I have used the Qwen 0.6B model as the Embedding model. As our system is a multilingual chatbot, where the knowledge base is in Bangla and we have to perform question-answering both in Bangla and English, we have to understand the texts in both language correctly. This is why I have used this multilingual Embedding model. Another good alternative is the BGE-m3 model, but as the previous model ranks higher in the Embedding leaderboard on different tasks especially retrieval, I have used the Qwen model as the final Embedding model.
How it captures meaning: A model like Qwen 0.6B captures the meaning of text by leveraging a Transformer-based neural network architecture. During its training on vast amounts of multilingual text, the model learns the complex statistical relationships, context, and nuances between words and sentences. When I provide it with a chunk of text, it converts that text into a high-dimensional numerical vector called an 'embedding'. The key to this process is that texts with similar semantic meanings will result in vectors that are mathematically 'close' to each other in this vector space. This allows the system to find relevant document chunks by comparing the vector of the user's query with the vectors of all the document chunks, simply by finding the ones that are closest.

Question: How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?
Answer: I am first creating the knowledge vectorstore for all the splitted chunks from the document using Qwen Embedding model. I use this vectorstore as the retriever directly when a new text query appears. As the similarity measure l am using cosine similarity based checking.I have initially chosen, the MMR-based approach but it wasn't correctly retrieving matched items.
I have configured the retriever with k=50 and search type was set to "similarity".
k=50: This instructs the retriever to return a total of 50 chunks. This provides a substantial amount of context for the LLM to form a comprehensive answer.

As the vectorstore, I have used the faiss-cpu as this is the fastest one while performing retrieval operations, also the setup and knowledge base building process with it is simple and easier.

Question: How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?
Answer: To understand our document semantically better in multilingual context, I have used the Qwen 0.6B Embedding model. When a user query comes, it also first gets converted into an embedded vector using the same Qwen 0.6B model. This ensures both the knowledge base and user query gets understood in the same semantical way. Finally, each document chunk gets semantically compared with the vectorized user query and top K =50 matched chunks are returned.
I have provided the system prompt in such way that , the final answer will be grounded only according to user-query, previous chat history and the extracted matched context. In case, the context is empty, signaling probably the query is not related to our knowledge base, I have prompted to return a default response ‘sorry, I am unable to answer the question’

I ensure a meaningful comparison between the question and documents by using a shared semantic space.
First, all document chunks are converted into vector embeddings using the Qwen embedding model and stored.
When a user asks a question, their query is converted into a vector using the exact same Qwen model.
Because both the query and the documents are processed by the same model, their vectors exist in the same high-dimensional space, allowing for a meaningful semantic comparison using cosine similarity.
For handling problematic queries, I differentiate between two cases:
What happens if the query is missing context (is off-topic)? If a user's query is completely unrelated to the knowledge base, the cosine similarity based retrieval process will likely fail to find any relevant chunks with a similarity score above a certain threshold. My system checks if the retrieved context is empty. If it is, this signals an out-of-domain question, and the system is prompted to return a default response like, ‘I'm sorry, but my knowledge is limited to the provided document. I am unable to answer that question.’
What happens if the query is vague? A vague query (e.g., "What happened next?") might still retrieve some chunks, but they may not be the most relevant ones. The system's prompt forces the LLM to answer based only on this potentially low-quality context, which could lead to a generic or unhelpful answer. A more advanced version of this system could improve this by checking the confidence scores of the retrieved chunks. If the scores are low, indicating ambiguity, the system could be prompted to ask the user for clarification instead of attempting to answer."

Question: Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?
Answer: yes, the result seems relevant.As the chunking, Embedding models are chosen in optimal way, the system returns relevant nice answers. Though the retrieved context can be further improved if a reranker model is used on retrieved contexts, to ensure the answering LLM gets more relevant and concise context to build it's answer upon. As adding this reranker will add some more latency and our knowledge base is a small one, I have opted not to use it.
