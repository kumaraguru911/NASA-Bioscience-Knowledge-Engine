export async function runPipeline(query, fetchNew = false) {
  const API_URL = "http://localhost:5000/api/query"; // change if your backend differs

  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, fetchNew }),
  });

  if (!response.ok) {
    throw new Error(`Server returned ${response.status}`);
  }

  const data = await response.json();
  return data.results || data;
}
