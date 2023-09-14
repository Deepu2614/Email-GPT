import 'dotenv/config';
import Nylas from "nylas"
import OpenAI from "openai";
import { PineconeClient } from '@pinecone-database/pinecone';

const pinecone = new PineconeClient();

await pinecone.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: 'gcp-starter'
});

const list = await pinecone.listIndexes();
// setup the pinecone vector index
const index = pinecone.Index('email-gpt-2');
console.log('index name', { list });

Nylas.config({
    clientId: process.env.CLIENT_ID,
    clientSecret: process.env.CLIENT_SECRET,
});

const nylas = Nylas.with(process.env.ACCESS_TOKEN);

const openai = new OpenAI({
    apiKey: process.env.OPEN_AI_API_KEY,
});

const currentAccount = await nylas.account.get();

console.log('current account', { currentAccount })

// grab an email
const retrieveMessages = async () => {
  const messages = await nylas.messages.list({ limit: 1 });
  const message = messages[0];
  
  console.log('message', { message });
  
  const id = message.id as string;
  
  const metadata = {
    snippet: message.snippet,
    fromName: message?.from?.[0].name,
    fromEmail: message.from?.[0].email,
    toName: message.to[0].name,
    toEmail: message.to[0].email,
    subject: message.subject,
    date: message.date,
    id: message.id,
  };
  
  console.log('message metadata', { metadata })
  return { message, metadata, id };
}

const createEmbeddings = async (message: any) => {
  const embeddings = await openai.embeddings.create({
    input: message.snippet as string,
    model: 'text-embedding-ada-002'
  });
  
  // generate embeddings to store in pinecone
  const embedding = embeddings.data[0].embedding
  
  return embedding;
}

const storeVector = async (embedding: any, { id, metadata }: any) => {
  const upsertResponse = await index.upsert({
    upsertRequest: {
      vectors: [
        {
          id,
          values: embedding,
          metadata,
        },
      ],
    },
  });

  console.log( { upsertResponse } );

  // get the index stats
  const indexStats = await index.describeIndexStats({
    describeIndexStatsRequest: {},
  });

  console.log({ indexStats });
}

const createQueryVector = async (query: string) => {
  // encode queries using the same encoder model to create a query vector
  const queryEmbeddings = await openai.embeddings.create({
    input: userQuery,
    model: 'text-embedding-ada-002',
  });
  
  // Retrieve the generated query vector
  const queryEmbedding = queryEmbeddings.data[0].embedding;

  return queryEmbedding;
}

const contextSearch = async (queryEmbedding: any) => {
  // search the pinecone index
const searchResponse = await index.query({
  queryRequest: {
    vector: queryEmbedding, 
    topK: 2, 
    includeMetadata: true
  },
});

  // log out the possible search responses
  console.log({ searchResponse });

  const matches = searchResponse?.matches;

  // retrieve the contexts from the matches
  const contexts = matches ? matches.map((result: any) => result.metadata.snippet) : [];

  console.log({ contexts }, contexts[0]);

  return contexts;
}

const createPrompt = async (contexts: string[], userQuery: string) => {
  // create the prompts
  const promptStart = `
  Answer the question based no the context below. \n\n
  Context: \n`

  const promptEnd = `  
  Question: ${userQuery} \n
  Answer:`

  let joinedContexts = "\n\n---\n\n";

  const limit = 3750;

  // join the contexts together
  contexts.map((context, i) => {
  if(joinedContexts.length + context.length >= limit) {
    console.log('total length:', joinedContexts.length + context.slice(0,i).length);
    return;
  }

  joinedContexts = joinedContexts + context + "\n\n---\n\n";

  return;
  })

  // create the final prompt
  const queryWithContexts = promptStart + joinedContexts + promptEnd;

  console.log(queryWithContexts);

  return queryWithContexts;
}

const sendPrompt = async(prompt:string) => {
  // generate the answer
  const completion = await openai.completions.create({
    model: 'text-davinci-003',
    prompt,
    max_tokens: 150,
    top_p: 1,
  });
  
  console.log(completion.choices[0].text);

  return completion.choices[0].text;
}

// Retrieve emails using the Nylas Email API
const messageData = await retrieveMessages();

// Create a vector embedding for the email using the message.snippet
const embedding = await createEmbeddings(messageData.message);

// Store the embedding in a vector database including parts of the email as metadata
await storeVector(embedding, messageData);

// Create a prompt to send to the embedding model first
const userQuery = 'Did Blag email me about a meeting?';

// Create the query embedding using the same encode model as in `createEmbeddings`
const queryEmbedding = await createQueryVector(userQuery);

// Retrieve relevant contexts from the vector database
const contexts = await contextSearch(queryEmbedding);

// Generate a prompt with the additional contexts to send to the completion model
const prompt = await createPrompt(contexts, userQuery);

// Generate a response using the modified prompt
const answer = await sendPrompt(prompt);

