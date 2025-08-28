variable "gcp_scp_key" {
    description = "GCP service account key file"
    type        = string
}

variable "gcp_project_id" {
    description = "GCP project ID"
    type        = string
}

variable "gcp_region" {
    description = "GCP Region"
    type        = string
    default     = "us-central1"
}

variable "gcp_zone" {
    description = "GCP Region"
    type        = string
    default     = "us-central1-a"
}

variable "team_members_emails" {
    description = "User emails for IAM roles for Vertex AI"
    type        = list(string)
}

variable "service_account" {
    description = "Service account"
    type        = string
}

variable "vm_machine_type" {
  description = "Machine type for the VM"
  type        = string
  default     = "e2-standard-4"
}