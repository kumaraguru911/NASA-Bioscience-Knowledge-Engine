import React from "react";

function Results({ results }) {
  if (!results) return null;
  if (Array.isArray(results) && results.length === 0) {
    return <div className="result-section"><p>No results found.</p></div>;
  }

  return (
    <div className="result-section">
      <h2>Query Results</h2>
      {Array.isArray(results) ? (
        results.map((res, index) => (
          <div key={index} className="result-card">
            <h3>{res.title || `Result ${index + 1}`}</h3>
            <p>{res.text || res.chunk || "No content available."}</p>
          </div>
        ))
      ) : (
        <pre>{JSON.stringify(results, null, 2)}</pre>
      )}
    </div>
  );
}

export default Results;
