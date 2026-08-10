{{- define "chap.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "chap.fullname" -}}
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

{{- define "chap.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "chap.labels" -}}
helm.sh/chart: {{ include "chap.chart" . }}
{{ include "chap.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.global.commonLabels }}
{{ toYaml . -}}
{{- end }}
{{- end }}

{{- define "chap.selectorLabels" -}}
app.kubernetes.io/name: {{ include "chap.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* Labels for a component. Expects a dict with `ctx` (root context) and `component`. */}}
{{- define "chap.componentLabels" -}}
{{ include "chap.labels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- with (index .ctx.Values .component).labels }}
{{ toYaml . -}}
{{- end }}
{{- end }}

{{/* Selector labels for a component. Expects a dict with `ctx` (root context) and `component`. */}}
{{- define "chap.componentSelectorLabels" -}}
{{ include "chap.selectorLabels" .ctx }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/* Service account name for a component. Expects a dict with `ctx` and `component`. */}}
{{- define "chap.componentServiceAccountName" -}}
{{- $values := index .ctx.Values .component }}
{{- if $values.serviceAccount.create }}
{{- default (printf "%s-%s" (include "chap.fullname" .ctx) .component) $values.serviceAccount.name }}
{{- else }}
{{- default "default" $values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "chap.database.host" -}}
{{- if .Values.db.enabled -}}
{{ include "chap.fullname" . }}-db-rw
{{- else -}}
{{ required "externalDatabase.host is required when db.enabled is false" .Values.externalDatabase.host }}
{{- end -}}
{{- end }}

{{- define "chap.database.port" -}}
{{- if .Values.db.enabled -}}
5432
{{- else -}}
{{ .Values.externalDatabase.port }}
{{- end -}}
{{- end }}

{{- define "chap.database.name" -}}
{{- if .Values.db.enabled -}}
{{ .Values.db.database }}
{{- else -}}
{{ .Values.externalDatabase.database }}
{{- end -}}
{{- end }}

{{- define "chap.database.secretName" -}}
{{- if .Values.db.enabled -}}
{{ default (printf "%s-db" (include "chap.fullname" .)) .Values.db.existingSecret }}
{{- else -}}
{{ default (printf "%s-db" (include "chap.fullname" .)) .Values.externalDatabase.existingSecret }}
{{- end -}}
{{- end }}

{{- define "chap.database.usernameKey" -}}
{{- if .Values.db.enabled -}}
username
{{- else -}}
{{ .Values.externalDatabase.secretKeys.username }}
{{- end -}}
{{- end }}

{{- define "chap.database.passwordKey" -}}
{{- if .Values.db.enabled -}}
password
{{- else -}}
{{ .Values.externalDatabase.secretKeys.password }}
{{- end -}}
{{- end }}

{{/* The valkey subchart derives its fullname from the release name and its chart name. */}}
{{- define "chap.valkey.fullname" -}}
{{- if .Values.valkey.fullnameOverride -}}
{{ .Values.valkey.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else if contains "valkey" .Release.Name -}}
{{ .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else -}}
{{ printf "%s-valkey" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end -}}
{{- end }}

{{- define "chap.valkey.host" -}}
{{- if .Values.valkey.enabled -}}
{{ include "chap.valkey.fullname" . }}
{{- else -}}
{{ required "externalValkey.host is required when valkey.enabled is false" .Values.externalValkey.host }}
{{- end -}}
{{- end }}

{{- define "chap.valkey.port" -}}
{{- if .Values.valkey.enabled -}}
6379
{{- else -}}
{{ .Values.externalValkey.port }}
{{- end -}}
{{- end }}

{{- define "chap.valkey.secretName" -}}
{{- if .Values.valkey.enabled -}}
{{- if .Values.valkey.auth.usersExistingSecret -}}
{{ .Values.valkey.auth.usersExistingSecret }}
{{- else if (dig "auth" "aclUsers" "default" "password" "" .Values.valkey) -}}
{{ include "chap.valkey.fullname" . }}-auth
{{- else -}}
{{ fail "Set valkey.auth.aclUsers.default.password or valkey.auth.usersExistingSecret" }}
{{- end -}}
{{- else -}}
{{ default (printf "%s-valkey" (include "chap.fullname" .)) .Values.externalValkey.existingSecret }}
{{- end -}}
{{- end }}

{{- define "chap.valkey.passwordKey" -}}
{{- if .Values.valkey.enabled -}}
{{- if .Values.valkey.auth.usersExistingSecret -}}
{{ dig "auth" "aclUsers" "default" "passwordKey" "default" .Values.valkey }}
{{- else -}}
default-password
{{- end -}}
{{- else -}}
{{ .Values.externalValkey.secretKeys.password }}
{{- end -}}
{{- end }}

{{/* Environment variables shared by the api and worker for the PostgreSQL connection. */}}
{{- define "chap.databaseEnv" -}}
- name: POSTGRES_USER
  valueFrom:
    secretKeyRef:
      name: {{ include "chap.database.secretName" . }}
      key: {{ include "chap.database.usernameKey" . }}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "chap.database.secretName" . }}
      key: {{ include "chap.database.passwordKey" . }}
- name: POSTGRES_HOST
  value: {{ include "chap.database.host" . | quote }}
- name: POSTGRES_PORT
  value: {{ include "chap.database.port" . | quote }}
- name: POSTGRES_DB
  value: {{ include "chap.database.name" . | quote }}
- name: CHAP_DATABASE_URL
  value: "postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)"
{{- end }}

{{/* Environment variables shared by the api and worker for the Valkey connection. */}}
{{- define "chap.valkeyEnv" -}}
- name: REDIS_HOST
  value: {{ include "chap.valkey.host" . | quote }}
- name: REDIS_PORT
  value: {{ include "chap.valkey.port" . | quote }}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "chap.valkey.secretName" . }}
      key: {{ include "chap.valkey.passwordKey" . }}
- name: CELERY_BROKER
  value: "redis://:$(REDIS_PASSWORD)@$(REDIS_HOST):$(REDIS_PORT)/0"
{{- end }}
