import React, { useState, useEffect, useCallback, useRef } from "react";
import { marked } from "marked";

// Simple ChatInstance interface (no Carbon dependency)
interface ChatInstance {
  messaging: {
    addMessage?: (message: any) => Promise<void>;
    addMessageChunk?: (chunk: any) => void;
  };
  on?: (options: { type: string; handler: (event: any) => void }) => void;
}
import "./CardManager.css";
import "./CustomResponseStyles.css";
// Import components from CustomResponseExample
import TaskStatusDashboard from "./task_status_component";
import ActionStatusDashboard from "./action_status_component";
import CoderAgentOutput from "./coder_agent_output";
import AppAnalyzerComponent from "./app_analyzer_component";
import TaskDecompositionComponent from "./task_decomposition";
import ShortlisterComponent from "./shortlister";
import SingleExpandableContent from "./generic_component";
import ActionAgent from "./action_agent";
import QaAgentComponent from "./qa_agent";
import { FollowupAction } from "./Followup";
import { fetchStreamingData } from "./StreamingWorkflow";
import ToolCallFlowDisplay from "./ToolReview";

interface Step {
  id: string;
  title: string;
  content: string;
  expanded: boolean;
  isNew?: boolean;
  timestamp: number;
  completed?: boolean;
}

// Color constant for highlighting important information
const HIGHLIGHT_COLOR = "#4e00ec";

interface CardManagerProps {
  chatInstance: ChatInstance;
}

// Extend the global interface typing to include the new loader API
declare global {
  interface Window {
    aiSystemInterface?: {
      addStep: (title: string, content: string) => void;
      getAllSteps: () => Step[];
      stopProcessing: () => void;
      isProcessingStopped: () => boolean;
      setProcessingComplete: (isComplete: boolean) => void;
      forceReset: () => void;
      hasStepWithTitle: (title: string) => boolean;
      showNextCardLoader?: (show: boolean) => void;
    };
    CUGA_DEBUG_LOADERS?: boolean;
  }
}

const CardManager: React.FC<CardManagerProps> = ({ chatInstance }) => {
  const [currentSteps, setCurrentSteps] = useState<Step[]>([]);
  const [currentCardId, setCurrentCardId] = useState<string | null>(null);
  const [isProcessingComplete, setIsProcessingComplete] = useState(false);
  const [showDetails, setShowDetails] = useState<{ [key: string]: boolean }>({});
  const [isReasoningCollapsed, setIsReasoningCollapsed] = useState(false);
  const [hasFinalAnswer, setHasFinalAnswer] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isStopped, setIsStopped] = useState(false);
  const [viewMode, setViewMode] = useState<'inplace' | 'append'>('inplace');
  const [globalVariables, setGlobalVariables] = useState<Record<string, any>>({});
  const [variablesHistory, setVariablesHistory] = useState<Array<{
    id: string;
    title: string;
    timestamp: number;
    variables: Record<string, any>;
  }>>([]);
  const [selectedAnswerId, setSelectedAnswerId] = useState<string | null>(null);
  // Loader for next step within this card is derived from processing state
  const cardRef = useRef<HTMLDivElement>(null);
  const stepRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});

  // Function to mark a step as completed
  const markStepCompleted = useCallback((stepId: string) => {
    setCurrentSteps(prev =>
      prev.map(step =>
        step.id === stepId ? { ...step, completed: true } : step
      )
    );
  }, []);

  // Initialize global interface

  // No cross-card loader logic needed; loader will be shown within the card while processing

  useEffect(() => {
    if (typeof window !== "undefined") {
      console.log("Setting up global aiSystemInterface");
      window.aiSystemInterface = {
        addStep: (title: string, content: string) => {
          console.log("🎯 addStep called:", title, content);
          console.log("🎯 Content type:", typeof content);
          console.log("🎯 Current steps before adding:", currentSteps.length);
          
          // If content is JSON string, try to parse and log it
          if (typeof content === 'string' && (content.startsWith('{') || content.startsWith('['))) {
            try {
              const parsed = JSON.parse(content);
              console.log("🎯 Parsed content:", parsed);
              console.log("🎯 Has variables:", !!parsed.variables);
              console.log("🎯 Variables keys:", parsed.variables ? Object.keys(parsed.variables) : []);
            } catch (e) {
              console.log("🎯 Failed to parse content as JSON");
            }
          }
          
          const newStep: Step = {
            id: `step-${Date.now()}-${Math.random()}`,
            title,
            content,
            expanded: true,
            isNew: true,
            timestamp: Date.now(),
          };

          setCurrentSteps(prev => {
            console.log("🎯 setCurrentSteps called with prev length:", prev.length);
            // If this is the first step, start a new card
            if (prev.length === 0) {
              const newCardId = `card-${Date.now()}`;
              setCurrentCardId(newCardId);
              console.log("🎯 First step - creating new card:", newCardId);
              return [newStep];
            }
            // Otherwise, add to current card
            console.log("🎯 Adding to existing card");
            return [...prev, newStep];
          });

          // Handle in-place card switching vs append mode
          if (viewMode === 'inplace') {
          if (currentSteps.length > 0) {
            setCurrentStepIndex(prev => prev + 1);
          } else {
            setCurrentStepIndex(0);
            }
          }

          // Auto-expand "Waiting for your input" components and collapse reasoning
          if (title === "SuggestHumanActions") {
            setShowDetails(prev => ({
              ...prev,
              [newStep.id]: true
            }));
            // Collapse reasoning process when user action is needed
            setIsReasoningCollapsed(true);
          }

          // Check if this is a final answer step
          if (title === "FinalAnswerAgent" || title === "FinalAnswer" || title === "Answer") {
            console.log("🎯 Final answer detected, triggering reasoning collapse");
            setHasFinalAnswer(true);
            // Collapse reasoning immediately when final answer arrives
            setIsReasoningCollapsed(true);
            // Show details by default for final answer
            setShowDetails(prev => ({
              ...prev,
              [newStep.id]: true
            }));
            
            // Emit event to notify parent that final answer is complete
            setTimeout(() => {
              const event = new CustomEvent('finalAnswerComplete', {
                detail: { stepId: newStep.id }
              });
              window.dispatchEvent(event);
              console.log("🎯 Emitted finalAnswerComplete event");
            }, 500);
          }
        },
        // No external loader toggle needed for within-card loading
        getAllSteps: () => currentSteps,
        stopProcessing: () => {
          setIsStopped(true);
          setIsProcessingComplete(true);
          setIsReasoningCollapsed(true);
          setShowDetails({});
        },
        isProcessingStopped: () => isProcessingComplete,
        setProcessingComplete: (isComplete: boolean) => {
          setIsProcessingComplete(isComplete);
        },
        forceReset: () => {
          setCurrentSteps([]);
          setIsProcessingComplete(false);
          setCurrentCardId(null);
          setIsReasoningCollapsed(false);
          setHasFinalAnswer(false);
          setCurrentStepIndex(0);
          setIsStopped(false);
          setShowDetails({});
          stepRefs.current = {};
          // Note: variablesHistory is preserved across conversations
        },
        hasStepWithTitle: (title: string) => {
          return currentSteps.some(step => step.title === title);
        },
      };
    }
  }, [currentSteps, currentCardId, isProcessingComplete, viewMode]);


  // Auto-scroll to latest step
  useEffect(() => {
    if (currentSteps.length > 0) {
      const timeoutId = setTimeout(() => {
        const latestStep = currentSteps[currentSteps.length - 1];
        const latestStepRef = stepRefs.current[latestStep.id];
        
        if (latestStepRef) {
          latestStepRef.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
        } else if (cardRef.current) {
          // Fallback to container scroll if step ref not found
          cardRef.current.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
        }
      }, 100);
      
      return () => clearTimeout(timeoutId);
    }
  }, [currentSteps.length]);

  // Cleanup step refs on unmount
  useEffect(() => {
    return () => {
      stepRefs.current = {};
    };
  }, []);

  // Extract variables from final answer steps and track by turn
  useEffect(() => {
    console.log('[Variables Debug] Processing steps, total:', currentSteps.length);
    const newHistory: Array<{
      id: string;
      title: string;
      timestamp: number;
      variables: Record<string, any>;
    }> = [];
    
    let turnNumber = 0;
    
    currentSteps.forEach((step) => {
      console.log('[Variables Debug] Step:', step.title, 'Type:', typeof step.content);
      
      // Only process Answer or FinalAnswerAgent steps
      if (step.title !== "Answer" && step.title !== "FinalAnswerAgent") {
        return;
      }
      
      console.log('[Variables Debug] Processing Answer/FinalAnswerAgent step');
      
      try {
        let parsedContent: any;
        let variables: Record<string, any> = {};
        
        if (typeof step.content === "string") {
          try {
            parsedContent = JSON.parse(step.content);
            console.log('[Variables Debug] Parsed JSON content:', parsedContent);
            
            // Check if we have variables in the parsed content
            if (parsedContent.data !== undefined && parsedContent.variables) {
              variables = parsedContent.variables;
              console.log('[Variables Debug] Found variables in data:', variables);
            } else if (parsedContent.variables) {
              variables = parsedContent.variables;
              console.log('[Variables Debug] Found variables directly:', variables);
            }
          } catch (e) {
            console.log('[Variables Debug] Failed to parse JSON:', e);
          }
        } else if (step.content && typeof step.content === "object" && 'variables' in step.content) {
          const contentWithVars = step.content as { variables?: Record<string, any> };
          if (contentWithVars.variables) {
            variables = contentWithVars.variables;
            console.log('[Variables Debug] Found variables in object:', variables);
          }
        }
        
        // Only add to history if this step has variables
        if (Object.keys(variables).length > 0) {
          console.log('[Variables Debug] Adding to history with', Object.keys(variables).length, 'variables');
          newHistory.push({
            id: step.id,
            title: `Turn ${turnNumber}`,
            timestamp: step.timestamp,
            variables: variables
          });
          turnNumber++;
        } else {
          console.log('[Variables Debug] No variables found in this step');
        }
      } catch (e) {
        console.log('[Variables Debug] Error processing step:', e);
      }
    });
    
    // Update history only if it actually changed
    setVariablesHistory(prev => {
      // Check if history actually changed
      if (prev.length !== newHistory.length) {
        console.log('Variables history updated: length changed', prev.length, '->', newHistory.length);
        return newHistory;
      }
      
      // Check if any entries are different
      const hasChanges = prev.some((entry, index) => {
        const newEntry = newHistory[index];
        return !newEntry || 
               entry.id !== newEntry.id || 
               JSON.stringify(entry.variables) !== JSON.stringify(newEntry.variables);
      });
      
      if (hasChanges) {
        console.log('Variables history updated: content changed');
      }
      
      return hasChanges ? newHistory : prev;
    });
    
    // Update selectedAnswerId based on available history
    setSelectedAnswerId(currentSelectedId => {
      // If we have new history from current steps, use that
      if (newHistory.length > 0) {
        if (currentSelectedId && newHistory.find(e => e.id === currentSelectedId)) {
          // Keep current selection if it still exists in new history
          return currentSelectedId;
        }
        // Auto-select most recent from new history
        console.log('Auto-selecting most recent turn:', newHistory[newHistory.length - 1].title);
        return newHistory[newHistory.length - 1].id;
      }

      // No new history from current steps, check if we have existing history
      // This happens when forceReset is called - we want to preserve selection
      if (variablesHistory.length > 0) {
        if (currentSelectedId && variablesHistory.find(e => e.id === currentSelectedId)) {
          // Keep current selection if it exists in existing history
          return currentSelectedId;
        }
        // Auto-select most recent from existing history
        console.log('Preserving selection from existing history:', variablesHistory[variablesHistory.length - 1].title);
        return variablesHistory[variablesHistory.length - 1].id;
      }

      // No history at all
      return null;
    });
  }, [currentSteps]);
  
  // Update globalVariables based on selected answer
  useEffect(() => {
    if (selectedAnswerId) {
      const selected = variablesHistory.find(e => e.id === selectedAnswerId);
      if (selected) {
        setGlobalVariables(selected.variables);
      }
    } else if (variablesHistory.length > 0) {
      // Default to most recent
      setGlobalVariables(variablesHistory[variablesHistory.length - 1].variables);
    } else {
      setGlobalVariables({});
    }
  }, [selectedAnswerId, variablesHistory]);

  // Emit variables updates to App.tsx
  useEffect(() => {
    const event = new CustomEvent('variablesUpdate', {
      detail: {
        variables: globalVariables,
        history: variablesHistory
      }
    });
    window.dispatchEvent(event);
  }, [globalVariables, variablesHistory]);

  // Function to generate natural language descriptions for each case
  const getCaseDescription = (stepTitle: string, parsedContent: any) => {
    switch (stepTitle) {
      case "PlanControllerAgent":
        if (parsedContent.subtasks_progress && parsedContent.next_subtask) {
          const completed = parsedContent.subtasks_progress.filter((status: string) => status === "completed").length;
          const total = parsedContent.subtasks_progress.length;

          if (total === 0) {
            return `I'm managing the overall task progress. There's <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">one next task</span>. ${parsedContent.conclude_task ? 'The task is ready to be concluded.' : `Next up: <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${parsedContent.next_subtask}</span>`}`;
          }

          return `I'm managing the overall task progress. Currently <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${completed} out of ${total} subtasks</span> are completed. ${parsedContent.conclude_task ? 'The task is ready to be concluded.' : `Next up: <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${parsedContent.next_subtask}</span>`}`;
        }
        return "I'm analyzing the task structure and planning the execution approach.";
      
      case "TaskDecompositionAgent":
        const taskCount = parsedContent.task_decomposition?.length || 0;
        return `I've broken down your request into <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${taskCount} manageable steps</span>. Each step is designed to work with specific applications and accomplish a specific part of your overall goal.`;
      
      case "APIPlannerAgent":
        if (parsedContent.action && (parsedContent.action_input_coder_agent || parsedContent.action_input_shortlisting_agent || parsedContent.action_input_conclude_task)) {
          const actionType = parsedContent.action;
          if (actionType === "CoderAgent") {
            return `I'm preparing to write code for you. The task involves: <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${parsedContent.action_input_coder_agent?.task_description || 'Code generation task'}</span>`;
          } else if (actionType === "ApiShortlistingAgent") {
            const taskDesc = parsedContent.action_input_shortlisting_agent?.task_description;
            if (taskDesc) {
              const preview = taskDesc.length > 60 ? taskDesc.substring(0, 60) + '...' : taskDesc;
              return `I'm analyzing available APIs, <span style="color:${HIGHLIGHT_COLOR}; font-weight:500;">${preview}</span>`;
            }
            return `I'm analyzing available APIs to find the best options for your request. This will help me understand what tools are available to accomplish your task.`;
          } else if (actionType === "ConcludeTask") {
            const taskDesc = parsedContent.action_input_conclude_task?.final_response;
            if (taskDesc) {
              const preview = taskDesc.length > 60 ? taskDesc.substring(0, 60) + '...' : taskDesc;
              return `I'm ready to provide you with the final answer based on all the work completed so far. <span style="color:${HIGHLIGHT_COLOR}; font-weight:500;">${preview}</span>`;
            }
            return `I'm ready to provide you with the final answer based on all the work completed so far.`;
          }
        }
        return "I'm reflecting on the code and planning the next steps in the workflow.";
      
      case "CodeAgent":
        if (parsedContent.code) {
          const codeLines = parsedContent.code.split('\n').length;
          const outputPreview = parsedContent.execution_output ? parsedContent.execution_output.substring(0, 50) + (parsedContent.execution_output.length > 50 ? '...' : '') : '';
          return `I've generated and executed <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${codeLines} lines of code</span> to accomplish your request. Here's a preview of the output: <span style="color:#10b981; font-family:monospace; background:#f0fdf4; padding:2px 4px; border-radius:3px; font-weight:500;">${outputPreview}</span>`;
        }
        return "I'm working on generating code for your request.";
      
      case "ShortlisterAgent":
        if (parsedContent.result) {
          const apiCount = parsedContent.result.length;
          const topResult = parsedContent.result[0];
          const topScore = topResult?.relevance_score || 0;
          const apiName = topResult?.name || topResult?.title || 'Unknown API';
          const truncatedName = apiName.length > 30 ? apiName.substring(0, 30) + '...' : apiName;
          return `I've analyzed and shortlisted <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${apiCount} relevant APIs</span> for your request. The top match is <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${truncatedName}</span> with a <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${Math.round(topScore * 100)}% relevance score</span>.`;
          
        }
        return "I'm analyzing available APIs to find the most relevant ones for your request.";
      
      case "TaskAnalyzerAgent":
        if (parsedContent && Array.isArray(parsedContent)) {
          const appNames = parsedContent.map(app => `<span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${app.name}</span>`).join(', ');
          return `I've identified <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${parsedContent.length} integrated applications</span> that can help with your request: ${appNames}. These apps are ready to be used in the workflow.`;
        }
        return "I'm analyzing the available applications to understand what tools we can use.";
      
      case "PlannerAgent":
        return `I'm planning the next action in the workflow. This involves determining the best approach to continue working on your request.`;
      
      case "QaAgent":
        if (parsedContent.name && parsedContent.answer) {
          return `I've analyzed the question "<span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${parsedContent.name}</span>" and provided a comprehensive answer with <span style="color:${HIGHLIGHT_COLOR}; font-weight: 600;">${parsedContent.answer.split(' ').length} words</span>.`;
        }
        return "I'm processing a question and preparing a detailed answer.";
      
      case "FinalAnswerAgent":
        if (parsedContent.final_answer) {
          return `I've completed your request and prepared the final answer.`;
        }
        return "I'm preparing the final answer to your request.";
      
      case "ReuseAgent":
        if (typeof parsedContent === "string")
          return parsedContent.split("\n")[0];
        return "Save and reuse operation completed."
      
      case "SuggestHumanActions":
        if (parsedContent.action_id) {
          return "I'm waiting for your input to continue. Please review the suggested action and let me know how you'd like to proceed.";
        }
        return "I'm preparing suggestions for your next action.";
      case "APICodePlannerAgent":
        const contentPreview = typeof parsedContent === 'string' ? parsedContent : JSON.stringify(parsedContent);
        const preview = contentPreview.length > 80 ? contentPreview.substring(0, 80) + '...' : contentPreview;
        return `I've generated a plan for the coding agent to follow. Plan preview: <span style="color:${HIGHLIGHT_COLOR}; font-weight:500;">${preview}</span>`;
      default:
        return "I'm processing your request and working on the next step in the workflow.";
    }
  };

  // Memoized function to render the appropriate component based on step title and content
  const renderStepContent = useCallback((step: Step) => {
    try {
      let parsedContent: any;

      if (typeof step.content === "string") {
        try {
          parsedContent = JSON.parse(step.content);
          const keys = Object.keys(parsedContent);
          
          console.log(`[${step.title}] Raw parsed content:`, parsedContent);
          console.log(`[${step.title}] Has data:`, parsedContent.data !== undefined);
          console.log(`[${step.title}] Has variables:`, !!parsedContent.variables);
          
          // Check if we have variables in the parsed content
          if (parsedContent.data !== undefined && parsedContent.variables) {
            console.log(`[${step.title}] Processing with variables...`);
            
            // For Answer step with variables: treat data as final_answer
            if (step.title === "Answer" || step.title === "FinalAnswerAgent") {
              parsedContent = {
                final_answer: parsedContent.data,
                variables: parsedContent.variables
              };
              console.log(`[${step.title}] Converted to final_answer format:`, parsedContent);
            } else if (typeof parsedContent.data === "object" && !Array.isArray(parsedContent.data)) {
              // Keep both data and variables if data is an object
              parsedContent = {
                ...parsedContent.data,
                variables: parsedContent.variables
              };
            } else {
              // If data is not an object, keep as is with variables
              parsedContent = {
                data: parsedContent.data,
                variables: parsedContent.variables
              };
            }
          } else if (keys.length === 1 && keys[0] === "data") {
            // Only data, no variables
            const data = parsedContent.data;
            parsedContent = data;
          }
        } catch (e) {
          parsedContent = step.content; // fallback
        }
      } else {
        parsedContent = step.content; // already an object
      }
      let outputElements = [];
      if (parsedContent && parsedContent.additional_data && parsedContent.additional_data.tool) {
        const newElem = <ToolCallFlowDisplay toolData={parsedContent.additional_data.tool} />;
        outputElements.push(newElem);
      }

      let mainElement = null;

      switch (step.title) {
        case "PlanControllerAgent":
          if (parsedContent.subtasks_progress && parsedContent.next_subtask) {
            mainElement = <TaskStatusDashboard taskData={parsedContent} />;
          }
          break;
        case "TaskDecompositionAgent":
          mainElement = <TaskDecompositionComponent decompositionData={parsedContent} />;
          break;
        case "APIPlannerAgent":
          if (
            parsedContent.action &&
            (parsedContent.action_input_coder_agent ||
              parsedContent.action_input_shortlisting_agent ||
              parsedContent.action_input_conclude_task)
          ) {
            mainElement = <ActionStatusDashboard actionData={parsedContent} />;
          } else {
            mainElement = <SingleExpandableContent title={"Code Reflection"} content={parsedContent} />;
          }
          break;
        case "CodeAgent":
          if (parsedContent.code) {
            mainElement = <CoderAgentOutput coderData={parsedContent} />;
          }
          break;
        case "ShortlisterAgent":
          if (parsedContent) {
            mainElement = <ShortlisterComponent shortlisterData={parsedContent} />;
          }
          break;
        case "WaitForResponse":
          return null;
        case "TaskAnalyzerAgent":
          if (parsedContent && Array.isArray(parsedContent)) {
            mainElement = <AppAnalyzerComponent appData={parsedContent} />;
          }
          break;
        case "PlannerAgent":
          if (parsedContent) {
            mainElement = <ActionAgent agentData={parsedContent} />;
          }
          break;
        case "simple_text_box":
          if (parsedContent) {
            mainElement = parsedContent;
          }
          break;
        case "QaAgent":
          if (parsedContent) {
            mainElement = <QaAgentComponent qaData={parsedContent} />;
          }
          break;
        case "Answer":
        case "FinalAnswerAgent":
          if (parsedContent) {
            // Handle both cases: final_answer field or just a string content
            const answerText = parsedContent.final_answer || (typeof parsedContent === 'string' ? parsedContent : null);
            
            console.log('Answer/FinalAnswerAgent - parsedContent:', parsedContent);
            console.log('Answer/FinalAnswerAgent - answerText:', answerText);
            
            if (answerText) {
              mainElement = (
                <div
                  style={{
                    fontSize: "14px",
                    lineHeight: "1.6",
                    color: "#1e293b"
                  }}
                  dangerouslySetInnerHTML={{ __html: marked(answerText) }}
                />
              );
            }
          }
          break;
        case "SuggestHumanActions":
          if (parsedContent && parsedContent.action_id) {
            mainElement = (
              <FollowupAction
                followupAction={parsedContent}
                callback={async (d: any) => {
                  console.log("calling fetch again");
                  // Mark this step as completed before proceeding
                  markStepCompleted(step.id);
                  await fetchStreamingData(chatInstance, "", d);
                }}
              />
            );
          }
          break;
        default:
          const isJSONLike =
            parsedContent !== null &&
            (typeof parsedContent === "object" || Array.isArray(parsedContent)) &&
            !(parsedContent instanceof Date) &&
            !(parsedContent instanceof RegExp);
          if (isJSONLike) {
            parsedContent = JSON.stringify(parsedContent, null, 2);
            parsedContent = `\`\`\`json\n${parsedContent}\n\`\`\``;
          }
          if (!parsedContent) {
            parsedContent = "";
          }
          mainElement = <SingleExpandableContent title={step.title} content={parsedContent} />;
      }

      // Add main element to outputElements if it exists
      if (mainElement) {
        outputElements.push(mainElement);
      }

      return <div>{outputElements}</div>;
    } catch (error) {
      console.log(`Failed to parse JSON for step ${step.title}:`, error);
      return null;
    }
  }, [chatInstance, markStepCompleted]);

  // Memoized button click handler
  const handleToggleDetails = useCallback((stepId: string) => {
    console.log('Button clicked for step:', stepId, 'Current state:', showDetails[stepId]);
    setShowDetails(prev => ({ ...prev, [stepId]: !prev[stepId] }));
  }, [showDetails]);

  // Handle reasoning collapse toggle
  const handleToggleReasoning = useCallback(() => {
    setIsReasoningCollapsed(prev => !prev);
  }, []);

  const mapStepTitle = (stepTitle: string) => {
    const titleMap = {
      TaskDecompositionAgent: "Decomposed task into steps",
      TaskAnalyzerAgent: "Analyzed available applications",
      PlanControllerAgent: "Controlled task execution",
      SuggestHumanActions: (
        <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div className="w-4 h-4 border-2 border-blue-200 border-t-blue-500 rounded-full animate-spin"></div>
          <span>Waiting for your input</span>
        </span>
      ),
      APIPlannerAgent: "Planned API actions",
      APICodePlannerAgent: "Planned steps for coding agent",
      CodeAgent: "Generated code solution",
      ShortlisterAgent: "Shortlisted relevant APIs",
      QaAgent: "Answered question",
      FinalAnswerAgent: "Completed final answer",
      Answer: "Answer",
    };

    return (titleMap as any)[stepTitle] || stepTitle;
  };

  console.log("CardManager render - currentSteps:", currentSteps.length, "isProcessingComplete:", isProcessingComplete);
  
  // Check if there's an error step
  const hasErrorStep = currentSteps.some(step => step.title === "Error");

  // Separate final answer steps and active user action steps from reasoning steps
  const finalAnswerSteps = currentSteps.filter(step =>
    step.title === "FinalAnswerAgent" || step.title === "FinalAnswer"
  );

  // Show SuggestHumanActions as active if it's not marked as completed
  const userActionSteps = currentSteps.filter(step =>
    step.title === "SuggestHumanActions" && !step.completed
  );

  // Include completed SuggestHumanActions in reasoning steps
  const reasoningSteps = currentSteps.filter(step =>
    step.title !== "FinalAnswerAgent" &&
    step.title !== "FinalAnswer" &&
    !(step.title === "SuggestHumanActions" && !step.completed)
  );

  // Get current step to display (before final answer or user action)
  const currentStep = currentSteps[currentStepIndex];
  const isShowingCurrentStep = !isStopped && viewMode === 'inplace' && !hasFinalAnswer && userActionSteps.length === 0 && currentStep;
  const isLoading = !isStopped && currentSteps.length > 0 && !isProcessingComplete && !hasFinalAnswer && userActionSteps.length === 0 && !hasErrorStep;
  
  // Helper function to render a single step card
  const renderStepCard = (step: Step, isCurrentStep: boolean = false) => {
    // Parse content for description
    let parsedContent;
    try {
      if (typeof step.content === "string") {
        try {
          parsedContent = JSON.parse(step.content);
          const keys = Object.keys(parsedContent);
          if (keys.length === 1 && keys[0] === "data") {
            const data = parsedContent.data;
            parsedContent = data;
          }
        } catch (e) {
          parsedContent = step.content;
        }
      } else {
        parsedContent = step.content;
      }
    } catch (error) {
      parsedContent = step.content;
    }

    if (step.title === "simple_text") {
      return <div key={step.id} style={{ marginBottom: "10px" }}>{step.content}</div>;
    }

    // Only render component content if details are shown
    const componentContent = showDetails[step.id] ? renderStepContent(step) : null;
    
    return (
      <div
        key={step.id}
        ref={(el) => {
          stepRefs.current[step.id] = el;
        }}
        className={`component-container ${step.isNew ? "new-component" : ""} ${isCurrentStep ? "current-step" : ""}`}
        style={{
          marginBottom: "16px",
          padding: "12px",
          paddingTop: "28px",
          backgroundColor: "#ffffff",
          borderRadius: "6px",
          border: "1px solid #e2e8f0",
          boxShadow: "0 1px 3px rgba(0, 0, 0, 0.05)",
          position: "relative",
        }}
      >
        {/* Component Title */}
        <div
          style={{
            marginBottom: "12px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <h3
            style={{
              fontSize: "14px",
              fontWeight: "500",
              color: "#475569",
              margin: "0",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            {mapStepTitle(step.title)}
          </h3>
        </div>

        {/* Natural Language Description */}
        <div
          style={{
            marginBottom: "12px",
          }}
        >
          <p
            style={{
              margin: "0",
              fontSize: "13px",
              color: "#64748b",
              lineHeight: "1.4",
            }}
            dangerouslySetInnerHTML={{ __html: getCaseDescription(step.title, parsedContent) }}
          />
        </div>

        {/* Component Content - Only show if showDetails is true */}
        {componentContent && (
          <div>{componentContent}</div>
        )}

        {/* Top-right details toggle */}
          <button
            onClick={() => handleToggleDetails(step.id)}
            style={{
            position: "absolute",
            right: "8px",
            top: "8px",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            background: "transparent",
            border: "1px solid #e5e7eb",
            borderRadius: "12px",
              padding: "4px 8px",
              fontSize: "11px",
            color: showDetails[step.id] ? "#3b82f6" : "#64748b",
              cursor: "pointer",
            }}
            onMouseOver={(e) => {
            (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#f8fafc";
            }}
            onMouseOut={(e) => {
            (e.currentTarget as HTMLButtonElement).style.backgroundColor = "transparent";
          }}
        >
          <span
            style={{
              display: "inline-block",
              transform: showDetails[step.id] ? "rotate(180deg)" : "rotate(0deg)",
              transition: "transform 0.2s ease",
              fontSize: "12px",
            }}
          >
            ▼
          </span>
          <span>details</span>
          </button>
        </div>
    );
  };

  return (
    <div 
      className="components-container" 
      ref={cardRef}
    >
      {/* View mode toggle */}
      {!isStopped && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "6px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "11px", color: "#64748b" }}>View:</span>
            <button
              onClick={() => setViewMode('inplace')}
              style={{
                padding: "2px 6px",
                backgroundColor: viewMode === 'inplace' ? "#2563eb" : "transparent",
                color: viewMode === 'inplace' ? "#ffffff" : "#64748b",
                border: "1px solid #e5e7eb",
                borderRadius: "3px",
                fontSize: "10px",
                fontWeight: 500,
                cursor: "pointer",
              }}
            >
              In-place
            </button>
            <button
              onClick={() => setViewMode('append')}
              style={{
                padding: "2px 6px",
                backgroundColor: viewMode === 'append' ? "#2563eb" : "transparent",
                color: viewMode === 'append' ? "#ffffff" : "#64748b",
                border: "1px solid #e5e7eb",
                borderRadius: "3px",
                fontSize: "10px",
                fontWeight: 500,
                cursor: "pointer",
              }}
            >
              Append
            </button>
          </div>
        </div>
      )}

      {/* Append mode */}
      {!isStopped && viewMode === 'append' && currentSteps.length > 0 && (
        hasFinalAnswer ? (
          <div>
            {/* Collapsed Reasoning wrapper with prior steps */}
            {reasoningSteps.length > 0 && (
              <div
                style={{
                  marginBottom: "16px",
                  padding: "12px",
                  backgroundColor: "#f8fafc",
                  borderRadius: "8px",
                  border: "1px solid #e2e8f0",
                  boxShadow: "0 1px 3px rgba(0, 0, 0, 0.05)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    cursor: "pointer",
                    userSelect: "none",
                  }}
                  onClick={handleToggleReasoning}
                >
                  <h3
                    style={{
                      fontSize: "16px",
                      fontWeight: "600",
                      color: "#374151",
                      margin: "0",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                    }}
                  >
                    <span
                      style={{
                        transform: isReasoningCollapsed ? "rotate(-90deg)" : "rotate(0deg)",
                        transition: "transform 0.3s ease",
                        fontSize: "14px",
                      }}
                    >
                      ▼
                    </span>
                    Reasoning Process
                    <span
                      style={{
                        fontSize: "12px",
                        fontWeight: "400",
                        color: "#6b7280",
                        backgroundColor: "#e5e7eb",
                        padding: "2px 8px",
                        borderRadius: "12px",
                      }}
                    >
                      {reasoningSteps.length} steps
                    </span>
                  </h3>
                  <div
                    style={{
                      fontSize: "12px",
                      color: "#6b7280",
                      fontStyle: "italic",
                    }}
                  >
                    {isReasoningCollapsed ? "Click to expand" : "Click to collapse"}
                  </div>
                </div>

                <div
                  style={{
                    maxHeight: isReasoningCollapsed ? "0" : "10000px",
                    overflow: "hidden",
                    transition: "max-height 0.5s ease-in-out, opacity 0.3s ease-in-out",
                    opacity: isReasoningCollapsed ? 0 : 1,
                  }}
                >
                  <div style={{ marginTop: "12px" }}>
                    {reasoningSteps.map((step) => renderStepCard(step, false))}
                  </div>
                </div>
              </div>
            )}

            {/* Final Answer card(s) */}
            {finalAnswerSteps.map((step) => renderStepCard(step, false))}
          </div>
        ) : (
          <div>
            {currentSteps.map((step) => (
              <div key={step.id}>{renderStepCard(step, false)}</div>
            ))}
          </div>
        )
      )}
      {/* When stopped, show a collapsed Reasoning section containing all steps */}
      {isStopped && currentSteps.length > 0 && (
        <div
          style={{
            marginBottom: "16px",
            padding: "12px",
            backgroundColor: "#f8fafc",
            borderRadius: "8px",
            border: "1px solid #e2e8f0",
            boxShadow: "0 1px 3px rgba(0, 0, 0, 0.05)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              cursor: "pointer",
              userSelect: "none",
            }}
            onClick={handleToggleReasoning}
          >
            <h3
              style={{
                fontSize: "16px",
                fontWeight: "600",
                color: "#374151",
                margin: "0",
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              <span
                style={{
                  transform: isReasoningCollapsed ? "rotate(-90deg)" : "rotate(0deg)",
                  transition: "transform 0.3s ease",
                  fontSize: "14px",
                }}
              >
                ▼
              </span>
              Reasoning Process
              <span
                style={{
                  fontSize: "12px",
                  fontWeight: "400",
                  color: "#6b7280",
                  backgroundColor: "#e5e7eb",
                  padding: "2px 8px",
                  borderRadius: "12px",
                }}
              >
                {currentSteps.length} steps
              </span>
            </h3>
            <div
              style={{
                fontSize: "12px",
                color: "#6b7280",
                fontStyle: "italic",
              }}
            >
              {isReasoningCollapsed ? "Click to expand" : "Click to collapse"}
            </div>
          </div>

          <div
            style={{
              maxHeight: isReasoningCollapsed ? "0" : "10000px",
              overflow: "hidden",
              transition: "max-height 0.5s ease-in-out, opacity 0.3s ease-in-out",
              opacity: isReasoningCollapsed ? 0 : 1,
            }}
          >
            <div style={{ marginTop: "12px" }}>
              {currentSteps.map((step) => renderStepCard(step, false))}
            </div>
          </div>
        </div>
      )}

      {/* Final outside card indicating interruption */}
      {isStopped && (
        <div style={{ marginTop: "8px" }}>
          <div
            style={{
              marginBottom: "16px",
              padding: "12px",
              backgroundColor: "#ffffff",
              borderRadius: "6px",
              border: "1px solid #e2e8f0",
              boxShadow: "0 1px 3px rgba(0, 0, 0, 0.05)",
            }}
          >
        <div
          style={{
            marginBottom: "12px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <h3
                style={{
                  fontSize: "14px",
                  fontWeight: "500",
                  color: "#475569",
                  margin: "0",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                Task Interrupted
              </h3>
            </div>
            <div>
          <p
            style={{
              margin: "0",
              fontSize: "13px",
              color: "#64748b",
              lineHeight: "1.4",
            }}
              >
                The task was stopped by the user.
              </p>
        </div>
      </div>
        </div>
      )}

      {/* Reasoning Section - Collapsible when final answer or user action is present */}
      {!isStopped && viewMode === 'inplace' && (hasFinalAnswer || userActionSteps.length > 0) && reasoningSteps.length > 0 && (
        <div
          style={{
            marginBottom: "16px",
            padding: "12px",
            backgroundColor: "#f8fafc",
            borderRadius: "8px",
            border: "1px solid #e2e8f0",
            boxShadow: "0 1px 3px rgba(0, 0, 0, 0.05)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              cursor: "pointer",
              userSelect: "none",
            }}
            onClick={handleToggleReasoning}
          >
            <h3
              style={{
                fontSize: "16px",
                fontWeight: "600",
                color: "#374151",
                margin: "0",
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              <span
                style={{
                  transform: isReasoningCollapsed ? "rotate(-90deg)" : "rotate(0deg)",
                  transition: "transform 0.3s ease",
                  fontSize: "14px",
                }}
              >
                ▼
              </span>
              Reasoning Process
              <span
                style={{
                  fontSize: "12px",
                  fontWeight: "400",
                  color: "#6b7280",
                  backgroundColor: "#e5e7eb",
                  padding: "2px 8px",
                  borderRadius: "12px",
                }}
              >
                {reasoningSteps.length} steps
              </span>
            </h3>
            <div
              style={{
                fontSize: "12px",
                color: "#6b7280",
                fontStyle: "italic",
              }}
            >
              {isReasoningCollapsed ? "Click to expand" : "Click to collapse"}
            </div>
          </div>
          
          <div
            style={{
              maxHeight: isReasoningCollapsed ? "0" : "10000px",
              overflow: "hidden",
              transition: "max-height 0.5s ease-in-out, opacity 0.3s ease-in-out",
              opacity: isReasoningCollapsed ? 0 : 1,
            }}
          >
            <div style={{ marginTop: "12px" }}>
              {reasoningSteps.map((step) => renderStepCard(step, false))}
            </div>
          </div>
        </div>
      )}

      {/* Current Step Display - Shows one step at a time with smooth transitions */}
      {!isStopped && viewMode === 'inplace' && isShowingCurrentStep && (
        <div
          className={`current-step-container ${isLoading ? "loading-border" : ""}`}
          style={{
            position: "relative",
            minHeight: "200px",
          }}
        >
          {renderStepCard(currentStep, true)}
        </div>
      )}

      {/* Final Answer Steps - Always visible (in-place mode) */}
      {!isStopped && viewMode === 'inplace' && finalAnswerSteps.map((step) => renderStepCard(step, false))}

      {/* User Action Steps - Always visible when present (in-place mode) */}
      {!isStopped && viewMode === 'inplace' && userActionSteps.map((step) => renderStepCard(step, false))}

      {/* Loading indicator - Only show when processing and no current step */}
      {(!isStopped && viewMode === 'inplace' && currentSteps.length > 0 && !isProcessingComplete && !hasFinalAnswer && userActionSteps.length === 0 && !hasErrorStep && !isShowingCurrentStep) && (
        <div style={{ marginTop: "8px", marginBottom: "2px" }}>
          <div
            style={{
              fontSize: "10px",
              color: "#94a3b8",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: "4px",
              userSelect: "none",
            }}
          >
            <span>CUGA is thinking..</span>
          </div>
          <div
            style={{
              height: "4px",
              position: "relative",
              overflow: "hidden",
              background: "#eef2ff",
              borderRadius: "9999px",
              boxShadow: "inset 0 0 0 1px #e5e7eb",
            }}
          >
            <div
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                bottom: 0,
                width: "28%",
                background: "linear-gradient(90deg, #a78bfa 0%, #6366f1 100%)",
                borderRadius: "9999px",
                animation: "cugaShimmer 1.7s infinite",
                boxShadow: "0 0 6px rgba(99,102,241,0.25)",
              }}
            />
          </div>
          <style>
            {`
              @keyframes cugaShimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(300%); }
              }
            `}
          </style>
        </div>
      )}

    </div>
  );
};

export default CardManager;