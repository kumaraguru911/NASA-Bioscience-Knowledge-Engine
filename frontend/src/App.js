import React, { useState } from "react";
import QueryForm from "./components/QueryForm";
import Results from "./components/Results";
import LoadingSpinner from "./components/LoadingSpinner";
import { runPipeline } from "./api/engineService";
//import nasaLogo from "./assets/logo.svg";

function App() {
  const [query, setQuery] = useState("");
  const [fetchNew, setFetchNew] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setResults(null);
    setError("");

    try {
      const data = await runPipeline(query, fetchNew);
      setResults(data);
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(err.message || "Something went wrong while fetching results.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        
        <h1>NASA Bioscience Knowledge Engine</h1>
      </header>

      <main>
        <QueryForm
          query={query}
          setQuery={setQuery}
          fetchNew={fetchNew}
          setFetchNew={setFetchNew}
          handleSubmit={handleSubmit}
          isLoading={isLoading}
        />

        {isLoading && <LoadingSpinner />}

        {error && (
          <div className="result-section error-box">
            <p>Error: {error}</p>
          </div>
        )}

        <Results results={results} />
      </main>
    </div>
  );
}

export default App;
