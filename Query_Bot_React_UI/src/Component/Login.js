import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./LoginTemplate.css";

const SoundComponent = () => {
  const playSound = () => {
    const audio = new Audio('/login-greeting.wav');
    audio.play()
      .then(_ => {
        // Audio started playing successfully
      })
      .catch(error => {
        console.error("Error playing audio:", error);
      });
  };

  useEffect(() => {
    // Play the sound when the component mounts
    playSound();

    // Event listener for browser refresh
    const handleRefresh = () => {
      playSound();
    };

    window.addEventListener('beforeunload', handleRefresh);

    return () => {
      // Cleanup the event listener
      window.removeEventListener('beforeunload', handleRefresh);
    };
  }, []);

  // Render nothing or a placeholder if needed
  return null;
};

const LoginTemplate = ({onLoginSuccess,status}) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    if (!status) {
      localStorage.removeItem("loginStatus");
    }
  }, [status]);

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const apiUrl = "http://localhost:5041/api/Login";
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ usermail: email, password: password }),
      });
  
      if (response.ok) {
        const data = await response.json();
        const isAdmin = data.isAdmin || false;
        const isPremium=data.isPremium || false;
        localStorage.setItem("premiumStatus",isPremium);
        localStorage.setItem("loginStatus",isAdmin)
        onLoginSuccess({isAdmin});

        navigate("/home", { state: { isAdmin } }); // Redirect on successful login

      } else if (response.status === 400) {
        setErrorMessage("UserMail and Password are required.");
      } else if (response.status === 401) {
        setErrorMessage("Invalid email or password");
      } else {
        setErrorMessage("An error occurred. Please try again.");
      }
    } catch (error) {
      setErrorMessage("An error occurred. Please try again.");
      console.error("Error:", error);
    }
  };

  return (
    <div className="bg">
      <SoundComponent />
      <section className="vh-100" style={{ height: "75%", width: "100%" }}>
        <div className="container py-3 h-75">
          <div className="row d-flex justify-content-center align-items-center h-100">
            <div className="col col-md-10 col-lg-7 carddiv">
              <div className="card ">
                <div className="row g-0">
                  <div className="col-md-7 col-lg-5 d-none d-md-block">
                    <img
                      src="./Query-Bot.jpeg"
                      alt="login form"
                      className="img-fluid"
                      style={{
                        borderRadius: "2rem 1rem 1rem 2rem",
                        height: "100%",
                        boxShadow: "0 4px 8px rgba(0, 0, 0, 0.9)",
                      }}
                    />
                  </div>
                  <div className="col-md-6 col-lg-7 d-flex align-items-center">
                    <div className="card-body p-4 p-lg-5 text-black">
                      <form onSubmit={handleLogin}>
                        <div className="d-flex align-items-center mb-3 pb-1">
                          <i
                            className="fas fa-cubes fa-2x me-3"
                            style={{ color: "#ff6219" }}
                          ></i>
                          <span className="h2 fw-bold mb-0">
                            Query Bot{" "}
                          </span>
                        </div>
                        <h5
                          className="fw-normal mb-3 pb-3"
                          style={{ letterSpacing: "1px" }}
                        >
                          Sign into your account
                        </h5>

                        <div className="form-outline mb-4">
                          <input
                            type="email"
                            id="form2Example17"
                            className="form-control form-control-lg"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                          />
                          <label
                            className="form-label"
                            htmlFor="form2Example17"
                          >
                            Email address
                          </label>
                        </div>

                        <div className="form-outline mb-4">
                          <input
                            type="password"
                            id="form2Example27"
                            className="form-control form-control-lg"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                          />
                          <label
                            className="form-label"
                            htmlFor="form2Example27"
                          >
                            Password
                          </label>
                        </div>

                        <div className="pt-1 mb-4">
                          <button
                            className="btn btn-dark btn-lg btn-block btn-zoom "
                            type="submit"
                          >
                            Login
                          </button>
                        </div>

                        {errorMessage && (
                          <div className="small text-danger">
                            {errorMessage}
                          </div>
                        )}

                        <a href="#!" className="small text-muted">
                          Terms of use.
                        </a>
                        <a href="#!" className="small text-muted">
                          Privacy policy
                        </a>
                      </form>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LoginTemplate;
