FROM maquins/ewars_plus_api:Upload

# CHAP patch (CLIM-617): replace foreach(.combine = rbind) at the CV-assembly
# step in Lag_Model_selection_ewars_By_District_api.R with dplyr::bind_rows so
# per-year frames with different _LAG{N} column names can be stacked.
# See: https://dhis2.atlassian.net/browse/CLIM-617
COPY --chown=app:app Lag_Model_selection_ewars_By_District_api.R /home/app/Lag_Model_selection_ewars_By_District_api.R
COPY --chown=app:app DBII_predictions_Vectorized_API.R /home/app/DBII_predictions_Vectorized_API.R
COPY --chown=app:app tests /home/app/tests

LABEL org.opencontainers.image.source="https://github.com/dhis2-chap/chap-core" \
      org.opencontainers.image.description="ewars_plus_api with CLIM-617 get_preds bind_rows fix on top of maquins/ewars_plus_api:Upload" \
      org.opencontainers.image.base.name="docker.io/maquins/ewars_plus_api:Upload"
