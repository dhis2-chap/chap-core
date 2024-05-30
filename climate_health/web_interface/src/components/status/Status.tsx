import React, { useEffect, useState } from 'react'
import { DefaultService } from '../../httpfunctions';
import styles from '../../styles/Status.module.css'

interface StatusProps {
  setIsReady : (v : boolean) => void
  upload : boolean
}

const Status = ({setIsReady, upload} : StatusProps) => {

  const [status, setStatus] = useState<undefined | {ready : boolean, status : string}>(undefined);
  const [errorMessage, setErrorMessage] = useState("");


  const getStatus = async () => {
    await DefaultService.getStatusStatusGet().catch((error : any) => {
      
      setErrorMessage("Could not get status")
      setIsReady(false)
      
    }).then((d : any) => {
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
    {!status?.ready &&

    <div className={status?.ready ? styles.ready : styles.notReady}>
      <div className={styles.statusContainer}>
        <div>Status:</div>
        <i>{status?.status == "idle" ? "Ready recive data" : status?.status}</i>
      </div>
    </div>
    }
    </>
  )
}

export default Status