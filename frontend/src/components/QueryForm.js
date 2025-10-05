import React from "react";

function QueryForm({ query, setQuery, fetchNew, setFetchNew, handleSubmit, isLoading }) {
  return (
    <form onSubmit={handleSubmit} className="query-form">
      <input
        type="text"
        value={query}
        placeholder="Enter your query (e.g., space nutrition and metabolism adaptation)"
        onChange={(e) => setQuery(e.target.value)}
        disabled={isLoading}
      />
      <div style={{ marginTop: "1rem" }}>
        <label>
          <input
            type="checkbox"
            checked={fetchNew}
            onChange={(e) => setFetchNew(e.target.checked)}
            disabled={isLoading}
          />
          Fetch new results
        </label>
      </div>
      <button type="submit" disabled={isLoading || !query}>
        {isLoading ? "Processing..." : "Run Query"}
      </button>
    </form>
  );
}

export default QueryForm;
