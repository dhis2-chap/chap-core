import React, { useState } from 'react'
import { DefaultService } from '../httpfunctions'
import { SendForm } from './SendForm'
import { error } from 'console'
import { Button } from '@mui/material'
import styles from '../styles/UploadForm.module.css'

interface UploadFormProps { 
  isReady : boolean
  setUpload : (e : boolean) => void
  upload : boolean
}

const UploadForm = ({isReady, setUpload ,upload} : UploadFormProps) => {
  //const [file, setFile] = useState<any>(null)
  const [file, setFile] = useState<any>(undefined)
  const [errorMessage, setErrorMessage] = useState<string>("")
  
  const handleFileSelect = (event : any) => {
    setFile(event.target.files[0])
  }

  const handleUpload = async () => {
    setUpload(true)
  }

  return (
    <div>
      <div className={styles.container}>   
        <div>
          <Button disabled={!isReady} disableElevation sx={{textTransform: 'none'}} variant="contained"
                component="label" >
                Select file (.zip)
                <input hidden type="file" onChange={handleFileSelect}  />
          </Button>
          <div className={styles.selectedFile}><i>{file?.name}</i></div>
        </div>
        <div>
          <Button disableElevation disabled={file == undefined || !isReady} variant='contained' sx={{textTransform: 'none'}} onClick={() => handleUpload()}>Make prediction</Button>          
        </div>
      
      </div>
      {upload && file && <SendForm formData={{file : file} as any} setErrorMessage={setErrorMessage} setUpload={setUpload}/>}
      <p className={styles.error}>{errorMessage}</p>
      
    </div>
  )
}

export default UploadForm