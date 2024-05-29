import React, { useState } from 'react';
import logo from './logo.svg';
import './App.css';
import { OpenAPI } from './httpfunctions';
import UploadForm from './components/UploadForm';
import Status from './components/status/Status';
import Result from './components/result/Result';

OpenAPI.BASE = 'http://localhost:8000'

function App() {

  const [isReady, setIsReady] = useState<boolean>(false)
  const [upload, setUpload] = useState<boolean>(false);

  return (
    <div className="App">
      <h2>CHAP</h2>
      
      
      <UploadForm upload={upload} setUpload={setUpload} isReady={isReady}/>

      <Status upload={upload} setIsReady={setIsReady}/>
      
      <Result isReady={isReady}/>
      
    </div>
  );
}

export default App;
