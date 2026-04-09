{{- define "chap-worker.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "chap-worker.fullname" -}}
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

{{- define "chap-worker.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "chap-worker.labels" -}}
helm.sh/chart: {{ include "chap-worker.chart" . }}
{{ include "chap-worker.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . -}}
{{- end }}
{{- end }}

{{- define "chap-worker.selectorLabels" -}}
app.kubernetes.io/name: {{ include "chap-worker.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "chap-worker.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "chap-worker.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "chap-worker.postgres.secretName" -}}
{{- if .Values.postgres.existingSecret }}
{{- .Values.postgres.existingSecret }}
{{- else }}
{{- include "chap-worker.fullname" . }}-postgres
{{- end }}
{{- end }}

{{- define "chap-worker.valkey.secretName" -}}
{{- if .Values.valkey.existingSecret }}
{{- .Values.valkey.existingSecret }}
{{- else }}
{{- include "chap-worker.fullname" . }}-valkey
{{- end }}
{{- end }}
