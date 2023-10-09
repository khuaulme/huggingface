import { HfInference } from "@huggingface/inference";
import dotenv from "dotenv";
dotenv.config();

const HF_ACCESS_TOKEN = process.env.HF_ACCESS_TOKEN;

const hf = new HfInference(HF_ACCESS_TOKEN);
const picture_url = "https://kwh-demos.s3.amazonaws.com/Karen2.JPG";
async function getImageBlob() {
  const imageResponse = await fetch(picture_url);

  const imageBlob = await imageResponse.blob();
  //   console.log(imageBlob);
  return imageBlob;
}

// const result = await hf.featureExtraction({
//   model: "sentence-transformers/distilbert-base-nli-mean-tokens",
//   inputs: "That is a happy person",
// });

// const miniLM = hf.endpoint(
//   "https://vnmiktbob2cu51s2.us-east-1.aws.endpoints.huggingface.cloud"
// );
const clipLM = hf.endpoint(
  "https://g9m2rh6h8t8wucei.us-east-1.aws.endpoints.huggingface.cloud"
);

try {
  const img = await getImageBlob();
  //   const embeddings = await hf.featureExtraction({
  //     inputs: { image: img },
  //   });

  const response = await hf.featureExtraction({
    model: "sentence-transformers/clip-ViT-B-32",
    inputs: { image: img },
  });

  //   console.log(response);
} catch (error) {
  console.error(error);
}

// { inputs: { image: Blob | ArrayBuffer }

// https://huggingface.co/docs/huggingface.js/main/en/inference/modules#textgenerationargs
