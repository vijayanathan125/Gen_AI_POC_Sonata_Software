// DocumentUpload.js

import React, { useState, useEffect } from "react";
import "./DocumentUpload.css";
import DeleteForeverIcon from "@mui/icons-material/DeleteForever";
import SideNav from "./SideNav";

const DocumentUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [documentList, setDocumentList] = useState([]);

  useEffect(() => {
    fetchDocumentList();
  }, []); 
  
  // Placeholder for fetchDocumentList
  const fetchDocumentList = async () => {
    try {
      const response = await fetch("http://localhost:5041/api/documents/list");
      if (response.ok) {
        const { files } = await response.json();
        setDocumentList(files.map((fileName) => ({ fileName })));
        fetchDocumentList();
      } else {
        console.error("Failed to fetch document list:", response.statusText);
      }
    } catch (error) {
      console.error("Error fetching document list:", error);
    }
  };
  

 // Fetch document list on component mount

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const response = await fetch(
          "http://localhost:5041/api/documents/upload",
          {
            method: "POST",
            body: formData,
          }
        );

        if (response.ok) {
          const uploadedDocument = await response.json();
          // Extract relevant properties from uploadedDocument
          const { fileName, size } = uploadedDocument;
          setDocumentList([...documentList, { fileName, size }]);

          setSelectedFile(null);

          // Clear the file input field
          const fileInput = document.getElementById("file-input");
          if (fileInput) {
            fileInput.value = "";
          }
        } else {
          console.error("Failed to upload document:", response.statusText);
        }
      } catch (error) {
        console.error("Error uploading document:", error);
      }
    }
  };

  const handleDelete = async (index, fileName) => {
    try {
      const response = await fetch(
        `http://localhost:5041/api/documents/delete/${fileName}`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        const updatedList = [...documentList];
        updatedList.splice(index, 1);
        setDocumentList(updatedList);
      } else {
        const errorMessage = await response.text(); // Get the error message from the response
        console.error(`Failed to delete document: ${errorMessage}`);
      }
    } catch (error) {
      console.error("Error deleting document:", error);
    }
  };

  console.log("documentList:", documentList);

  return (
    <div className="chatbox-container">
      <div className="side-nav-left">
        <SideNav />
      </div>
      <div className="document-upload-container">
        <h2>Document Upload</h2>
        <input type="file" onChange={handleFileChange} id="file-input" style={{backgroundColor:"white"}} />
        <button onClick={handleUpload}>Upload</button>

        <div className="document-list">
          <h3>Uploaded Documents</h3>
          <div className="table-container">
          <table className="document-table">
            <thead>
              <tr>
                <th>Document Name</th>
                {/* <th>Size</th> */}
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
                { documentList.map((document, index) => (
                <tr key={index}>
                  <td>{document.fileName}</td>
                  {/* <td>{document.size} bytes</td> */}
                  <td>
                  <DeleteForeverIcon onClick={() => handleDelete(index, document.fileName)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>

          <div className="doc-count">
            <p>Total Documents: {documentList.length}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentUpload;
