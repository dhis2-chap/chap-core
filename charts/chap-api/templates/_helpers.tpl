{{- define "chap-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "chap-api.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "chap-api.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "chap-api.labels" -}}
helm.sh/chart: {{ include "chap-api.chart" . }}
{{ include "chap-api.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . -}}
{{- end }}
{{- end }}

{{- define "chap-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "chap-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "chap-api.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "chap-api.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "chap-api.postgres.secretName" -}}
{{- if .Values.postgres.existingSecret }}
{{- .Values.postgres.existingSecret }}
{{- else }}
{{- include "chap-api.fullname" . }}-postgres
{{- end }}
{{- end }}

{{- define "chap-api.valkey.secretName" -}}
{{- if .Values.valkey.existingSecret }}
{{- .Values.valkey.existingSecret }}
{{- else }}
{{- include "chap-api.fullname" . }}-valkey
{{- end }}
{{- end }}
