import React, { useEffect, useState } from 'react'
import { DefaultService } from '../../httpfunctions';
import styles from '../../styles/Status.module.css'
import saveAs from 'file-saver';
import { LinearProgress } from '@mui/material';
import { read, stat } from 'fs';

interface StatusProps {
  setIsReady : (v : boolean) => void
  upload : boolean
}

const Status = ({setIsReady, upload} : StatusProps) => {

  const [status, setStatus] = useState<undefined | {ready : boolean, status : string}>(undefined);
  const [errorMessage, setErrorMessage] = useState("");


  const getStatus = async () => {
    let ready : boolean = status?.ready.valueOf() || false;
    //console.log(ready)
    await DefaultService.getStatusStatusGet().catch((error : any) => {
      
      setErrorMessage("Could not get status")
      setIsReady(false)
      
    }).then((d : any) => {
      //download result when the status change from false to true
      setErrorMessage("")
      setStatus(d)
      setIsReady(d?.ready);

      
    });
  }



  

  useEffect(() => {
    const interval = setInterval(() => {
      getStatus();
    }, 2000);

    getStatus(); // Call getStatus on init

    return () => {
      clearInterval(interval);
    };
  }, [upload]);
  


  return (
    <>
    <div className={styles.error}>{errorMessage}</div>
    {status && !status?.ready &&

    <div className={status?.ready ? styles.ready : styles.notReady}>
      <div className={styles.statusContainer}>
        <div>Status:</div>
        <i>{status?.status == "idle" ? "Ready recive data" : status?.status}</i>
      </div>
      <LinearProgress color='warning' />
    </div>
    }
    </>
  )
}

export default Status