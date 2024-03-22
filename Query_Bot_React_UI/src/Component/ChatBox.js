import React, { useState, useEffect, useRef } from "react";
import "./ChatBox.css";
import SendIcon from "@mui/icons-material/Send";
import axios from "axios";
import SideNav from "./SideNav";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";

const ChatBox = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false); // State for loading indicator
  const chatHistoryRef = useRef(null); // Reference to the chat history container
  const messageSoundRef = useRef(new Audio('/chat-message-sound.mp3'));
  //const[activeAPI,setActiveAPI]=useState("Context Setting")

  // useEffect(() => {
  //   // Retrieve active API from localStorage
  //   const storedAPI = localStorage.getItem("activeApi");
  //   if (storedAPI) {
  //     setActiveAPI(storedAPI);
  //   }
  // }, []);
  
  // Function to scroll the chat history container to the bottom
  const scrollToBottom = () => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom(); // Scroll to the bottom when messages update
  }, [messages]);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSendMessage = async () => {
    if (input.trim() === "") return;

    // Display user's query in the chat history
    const userMessage = {
      txt: input,
      type: "user",
    };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    // Play sound for user input
    messageSoundRef.current.play();

    // Set loading state to true while waiting for response
    setLoading(true);

    // let apiUrl;
    // if (activeAPI === "Context Setting") {
    //   apiUrl = "http://localhost:5005/context_query";
    // } 
    // else if (activeAPI === "BERT Method") {
    //   apiUrl = "http://localhost:5006/bert_query";
    // }

    // Send input to the Flask API
  // try {
  //      const response = await axios.post(apiUrl, {
  //      user_query: input,
  // });

     // Send input to the Flask API
    // try {
    //   const response = await axios.post(
    //     activeAPI === "BERT"
    //       ? "http://localhost:5002/bert_query"
    //       : "http://localhost:5001/context_query",
    //     {
    //       user_query: input,
    //     }
    //   );

    // Send input to the Flask API
    try {
      //const response = await axios.post("http://localhost:5001/context_query",
      const response = await axios.post("http://localhost:5002/bert_query",
      {
        user_query: input,
      });

      const systemResponse = {
        txt: response.data.response,
        type: "system",
      };
      setMessages((prevMessages) => [...prevMessages, systemResponse]);

      // Play sound for both user input and system response
      messageSoundRef.current.play();

    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = {
        txt: "Apologies for inconvenience, 😟 We ran into a problem. Please try again later.",
        type: "system",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }

    // Set loading state back to false after receiving response
    setLoading(false);

    // Clear the input field after sending the message
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault(); // Prevents the default behavior of the Enter key (e.g., new line in a textarea)
      handleSendMessage(); // Call the handleSendMessage function when Enter key is pressed
    }
  };

  // Function to scroll the chat history container to the bottom
  const scrollToEnd = () => {
    scrollToBottom();
  };

  // Function to scroll the chat history container to the top
  const scrollToStart = () => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = 0;
    }
  };

  return (
    <div className="chatbox-container">
      <div className="side-nav-left">
        <SideNav />
      </div>
      <div className="chat-container">
        <div className="chat-history" ref={chatHistoryRef}>
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              {message.type === "user" ? (
                <div className="user-message">{message.txt}</div>
              ) : (
                <div className="system-message">{message.txt}</div>
              )}
            </div>
          ))}
          {/* Loading indicator */}
          {loading && (
            <div className="center">
              {Array.from({ length: 10 }, (_, index) => (
            <div key={index} className="wave"></div>
            ))}
            </div>
          )}
        </div>
        <div className="chat-input">
          <input
            type="text"
            placeholder="Type your query..."
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
          />
          <button onClick={handleSendMessage}>
            <SendIcon />
          </button>
          <button onClick={scrollToStart}><ArrowUpwardIcon /></button>
          <button onClick={scrollToEnd}><ArrowDownwardIcon /></button>
        </div>
      </div>
    </div>
  );
};

export default ChatBox;