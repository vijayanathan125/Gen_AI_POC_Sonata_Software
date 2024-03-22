import "./SideNav.css";
import { NavLink, useLocation } from "react-router-dom";
import FileUploadIcon from "@mui/icons-material/FileUpload";
import ChatIcon from "@mui/icons-material/Chat";
import LogoutIcon from "@mui/icons-material/Logout";
import RoofingIcon from '@mui/icons-material/Roofing';
import React, { useEffect, useRef } from "react";
import "./SideNav.css";
//import React, { useState } from "react";
//import Switch from "@mui/material/Switch";

function SideNav({ isAdmin }) {
  const location = useLocation();
  const scrollSoundRef = useRef(new Audio('./scroll-sound.mp3')); // Adjust the path accordingly
  const clickSoundRef = useRef(new Audio('./button-click-sound.mp3')); // Adjust the path accordingly
  //const [activeAPI, setActiveAPI] = useState("Context Setting");

  // const toggleAPI = () => {
  //   setActiveAPI((prevAPI) =>
  //     prevAPI === "Context Setting" ? "BERT Method" : "Context Setting"
  //   );
  // };

  useEffect(() => {
    // Function to play sound on scroll
    const handleScroll = () => {
      scrollSoundRef.current.play();
    };

    // Add a scroll event listener
    window.addEventListener('scroll', handleScroll);

    // Clean up the event listener when the component unmounts
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const playClickSound = () => {
    clickSoundRef.current.play();
  };

  const status = localStorage.getItem("loginStatus");

  return (
    <div className="Sidebar">
      <div className="doc-logo">
        <h2 style={{"margin-left":"22%","marginTop":"5%"}}>Query Bot</h2>
      </div>

      <ul className="SidebarList">
       <li className="Row" id={location.pathname === "/home" ? "active" : ""}onClick={playClickSound}>
           <NavLink to="/home">
             <div id="icon">
               <RoofingIcon/>
             </div>
             <div id="title">Home</div>
           </NavLink>
         </li>
      </ul>

      <ul className="SidebarList">
        <li className="Row" id={location.pathname === "/chat" ? "active" : ""}onClick={playClickSound}>
          <NavLink to="/chat">
            <div id="icon">
              <ChatIcon></ChatIcon>
            </div>
            <div id="title">Chat</div>
          </NavLink>
        </li>

        {status === "true" && (
          <li
            className="Row"
            id={location.pathname === "/upload" ? "active" : ""}onClick={playClickSound}>
            <NavLink to="/upload">
              <div id="icon">
                <FileUploadIcon></FileUploadIcon>
              </div>
              <div id="title">Upload Document</div>
            </NavLink>
          </li>
        )}

        <li className="Row-logout" id="logout"onClick={playClickSound}>
          <NavLink to="/logout">
            <div id="icon">
              <LogoutIcon />
            </div>
            <div id="title">Logout</div>
          </NavLink>
        </li>

        {/* {<li>
          <Switch
            // checked={activeAPI === "BERT Method"}
            checked={localStorage.setItem("activeApi","BERT Method")}
            onChange={toggleAPI}
            inputProps={{ "aria-label": "Toggle API" }}
            disabled={localStorage.removeItem("activeApi")}
          />
          <p>Selected API: {activeAPI}</p>
        </li>} */}
      </ul>
    </div>
  );
}

export default SideNav;