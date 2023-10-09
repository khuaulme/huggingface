import { HfInference } from "@huggingface/inference";
import { MongoClient } from "mongodb";

import dotenv from "dotenv";
dotenv.config();
const uri = process.env.MONGODB_CONNECTION_STRING;

const HF_ACCESS_TOKEN = process.env.HF_ACCESS_TOKEN;

const hf = new HfInference(HF_ACCESS_TOKEN);

const client = new MongoClient(uri);

const getQueryEmbeddings = async (text) => {
  try {
    const embeddings = await hf.featureExtraction({
      model: "sentence-transformers/clip-ViT-B-32",
      inputs: text,
    });

    return embeddings;
  } catch (error) {
    console.error(error);
  }
};

const queryPosterEmbeddings = async (query) => {
  try {
    await client.connect();
    const db = client.db("sample_mflix");
    const collection = db.collection("movies");

    const results = await collection
      .aggregate([
        {
          $vectorSearch: {
            index: "vectorImageIndex",
            queryVector: await getQueryEmbeddings(query),
            path: "poster_embedding",
            numCandidates: 100,
            limit: 5,
          },
        },
        {
          $project: {
            _id: 0,
            title: 1,
            // plot:1,
            poster: 1,
          },
        },
      ])
      .toArray();

    console.log(results);
  } finally {
    console.log("Closing connection.");
    await client.close();
  }
};

const query = "alien invasion outer space";
queryPosterEmbeddings(query).catch(console.dir);

// const embeddings = await getQueryEmbeddings("cowboys on mountain");

// https://huggingface.co/docs/huggingface.js/main/en/inference/modules#textgenerationargs

/******************************************************************************
 * 
 *  const miniLM = hf.endpoint(
   "https://vnmiktbob2cu51s2.us-east-1.aws.endpoints.huggingface.cloud"
 );
 const clipLM = hf.endpoint(
   "https://g9m2rh6h8t8wucei.us-east-1.aws.endpoints.huggingface.cloud"
 );
 * 

 async function testConnection() {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log(
      "Pinged deployment. You successfully connected to your MongoDB Atlas cluster."
    );
  } finally {
    console.log("Closing connection.");
    await client.close();
  }
}
******************************************************************************** */
