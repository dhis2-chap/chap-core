import React, { useEffect } from 'react'
import { Body_post_zip_file_post_zip_file__post, DefaultService } from '../httpfunctions';
import { CircularProgress, LinearProgress } from '@mui/material';
import styles from '../styles/SendForm.module.css';

interface SendFormProps {
    formData : Body_post_zip_file_post_zip_file__post
    setErrorMessage : (message : string) => void
    setUpload : (uploaded : boolean) => void
}


export const SendForm = ({formData, setErrorMessage, setUpload} : SendFormProps) => {


  const sendFile = async () => {
    
    setErrorMessage("")
    await DefaultService.postZipFilePostZipFilePost(formData).catch((error : any) => {
      setErrorMessage(error?.body?.detail)
      setUpload(false) 
    }).then(() => {
      setUpload(false) 
    });
    
  }

  useEffect(() => {
    sendFile()
  }, [])
  

  
    
  return (
    <div className={styles.loading}>
     <LinearProgress />
    </div>
  )
}
