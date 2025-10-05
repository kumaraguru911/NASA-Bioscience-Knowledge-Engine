import React from "react";

function LoadingSpinner() {
  return (
    <div className="spinner-container">
      <div className="spinner"></div>
      <p>Fetching data from NASA’s knowledge vault...</p>
    </div>
  );
}

export default LoadingSpinner;
