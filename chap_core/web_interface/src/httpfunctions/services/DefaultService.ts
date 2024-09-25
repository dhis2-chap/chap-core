/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_post_zip_file_post_zip_file__post } from '../models/Body_post_zip_file_post_zip_file__post';
import type { State } from '../models/State';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class DefaultService {
    /**
     * Post Zip File
     * Post a zip file containing the data needed for training and evaluation, and start the training
     * @param formData
     * @returns any Successful Response
     * @throws ApiError
     */
    public static postZipFilePostZipFilePost(
        formData?: Body_post_zip_file_post_zip_file__post,
    ): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/post_zip_file/',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Results
     * Retrieve results made by the model
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getResultsGetResultsGet(): CancelablePromise<Array<Record<string, any>>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/get_results/',
        });
    }
    /**
     * Get Status
     * Retrieve the current status of the model
     * @returns State Successful Response
     * @throws ApiError
     */
    public static getStatusStatusGet(): CancelablePromise<State> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/status/',
        });
    }
}
