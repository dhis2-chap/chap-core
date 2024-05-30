import { Button } from '@mui/material'
import React, { useState } from 'react'
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import { DefaultService } from '../../httpfunctions';
import { saveAs } from 'file-saver';
import styles from '../../styles/Result.module.css'

interface ResultProps {
  isReady: boolean
}

const Result = ({ isReady }: ResultProps) => {

  const [errorMessage, setErrorMessage] = useState<string>("");

  const downloadResult = async () => {
    await DefaultService.getResultsGetResultsGet().catch((error: any) => {

    }).then((response: any) => {
      console.log(response)
      const blob = new Blob([JSON.stringify(response, null, 2)], { type: 'application/json' });
      const date = new Date().toJSON().slice(0, 10);
      saveAs(blob, `chap-result-${date}.json`);
    });
  }

  return (
    <div className={styles.buttonContainer}>
      <Button className={styles.result} color="success" variant='outlined' onClick={downloadResult} disableElevation sx={{ textTransform: 'none' }} endIcon={<FileDownloadIcon />}>Download latest result</Button>
      <span>{errorMessage}</span>
    </div>
  )
}

export default Result