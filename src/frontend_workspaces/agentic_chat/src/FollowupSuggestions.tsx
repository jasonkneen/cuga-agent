import React from "react";
import "./FollowupSuggestions.css";

interface FollowupSuggestionsProps {
  suggestions: string[];
  onSuggestionClick: (suggestion: string) => void;
}

export function FollowupSuggestions({ suggestions, onSuggestionClick }: FollowupSuggestionsProps) {
  if (suggestions.length === 0) {
    return null;
  }

  return (
    <div className="followup-suggestions-container">
      <div className="followup-suggestions-header">
        <span className="followup-suggestions-title">Suggested followup questions:</span>
      </div>
      <div className="followup-suggestions-list">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            className="followup-suggestion-chip"
            onClick={() => onSuggestionClick(suggestion)}
            type="button"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}

