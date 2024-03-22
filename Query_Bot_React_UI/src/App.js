import "./App.css";
import ChatBox from "./Component/ChatBox";
import DocumentUpload from "./Component/DocumentUpload";
import { BrowserRouter as Router, Route, Routes, Navigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";

import Login from "./Component/Login";
import { useState } from "react";
import SideNav from "./Component/SideNav";
import Home from "./Component/Home";
function App() {
  const [isAuthenticate, setIsAuthenticate] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  // const [activeAPI, setActiveAPI] = useState("Context Setting"); // Add this line

  const handleLogin = (adminstatus) => {
    setIsAuthenticate(true);
    setIsAdmin(adminstatus.isAdmin);
  };

  const ProtectedRoute = ({ element, path }) => {
    const isUploadRoute = path === "/upload";

    return isAuthenticate && (!isUploadRoute || isAdmin) ? (
      element
    ) : (
      <Navigate to="/" replace state={{ from: path }} />
    );
  };

  return (
    <div className="App">
      <Router>
        <Routes>
          <Route
            path="/"
            element={
              <Login
                onLoginSuccess={handleLogin}
                status={true}
              />
            }
          />
          {/* Define your protected routes using the ProtectedRoute component */}
          <Route
            path="/chat"
            element={<ProtectedRoute element={<ChatBox  />} path="/chat" />} // Modify this line
          />
           <Route
            path="/home"
            element={<ProtectedRoute element={<Home  />} path="/home" />} // Modify this line
          />
          <Route
            path="/upload"
            element={
              <ProtectedRoute element={<DocumentUpload />} path="/upload" />
            }
          />
          <Route
            path="/navbar"
            element={
              <ProtectedRoute
                element={<SideNav  isAdmin={isAdmin} />} // Modify this line
                path="/navbar"
              />
            }
          />
          <Route
            path="/logout"
            element={
              <ProtectedRoute
                element={
                  <Login
                    onLoginSuccess={handleLogin}
                    status={false}
                  />
                }
                path="/logout"
              />
            }
          />
        </Routes>
      </Router>
    </div>
  );
}

export default App;