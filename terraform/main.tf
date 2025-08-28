# Este código es compatible con Terraform 4.25.0 y versiones compatibles con 4.25.0.
# Para obtener información sobre la validación de este código de Terraform, consulta https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/google-cloud-platform-build#format-and-validate-the-configuration

#------------------------------------------------------------
#--------- Google Cloud Storage -----------------------------
#------------------------------------------------------------

resource "google_storage_bucket" "mlops-bucket" {
  name          = "mit-project-vertex-ai-artifacts"
  project       = var.gcp_project_id
  location      = var.gcp_region
  force_destroy = true

  uniform_bucket_level_access = true

  labels = {
    environment   = "dev"
    type = "storage"
    app = "mit-project"
  }

}

#------------------------------------------------------------
#--------- Iam Roles ----------------------------------------
#------------------------------------------------------------

# Se habilita la API de Cloud Resource Manager, necesaria para gestionar los roles de IAM.
resource "google_project_service" "cloudresourcemanager_api" {
  project            = var.gcp_project_id
  service            = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}

# Se habilita la API de IAM, necesaria para crear cuentas de servicio.
resource "google_project_service" "iam_api" {
  project            = var.gcp_project_id
  service            = "iam.googleapis.com"
  disable_on_destroy = false
}

# Asigna el rol de visor del project a los miembros del equipo
resource "google_project_iam_member" "viewer_team_members" {
  for_each = toset(var.team_members_emails)
  project  = var.gcp_project_id
  role     = "roles/viewer"
  member   = "user:${each.key}"
    depends_on = [
    google_project_service.cloudresourcemanager_api,
    google_project_service.iam_api
  ]
}

# Asigna el rol de usuario de Vertex AI a los miembros del equipo
resource "google_project_iam_member" "team_members_iam" {
  for_each = toset(var.team_members_emails)
  project  = var.gcp_project_id
  role     = "roles/aiplatform.user"
  member   = "user:${each.key}"
  depends_on = [
    google_project_service.cloudresourcemanager_api,
    google_project_service.iam_api
  ]
}

#------------------------------------------------------------
#--------- Vertex AI ----------------------------------------
#------------------------------------------------------------

# Verifica que la API de Vertex AI esté habilitada
resource "google_project_service" "vertex_ai_api" {
  project = var.gcp_project_id
  service = "aiplatform.googleapis.com"
  disable_on_destroy = false
  depends_on = [
    google_project_service.cloudresourcemanager_api,
    google_project_service.iam_api
  ]
}

# Crea una cuenta de servicio para Vertex AI Workbench
resource "google_service_account" "workbench_sa" {
  account_id   = "workbench-sa"
  display_name = "Service Account for Vertex AI Workbench"
  project      = var.gcp_project_id
}

# Asigna roles a la cuenta de servicio
resource "google_project_iam_member" "aiplatform_user" {
  project = var.gcp_project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.workbench_sa.email}"
  depends_on = [google_service_account.workbench_sa]
}

resource "google_project_iam_member" "storage_admin" {
  project = var.gcp_project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.workbench_sa.email}"
  depends_on = [google_service_account.workbench_sa]
}

resource "google_project_iam_member" "log_writer" {
  project = var.gcp_project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.workbench_sa.email}"
  depends_on = [google_service_account.workbench_sa]
}

resource "google_project_iam_member" "metric_writer" {
  project = var.gcp_project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.workbench_sa.email}"
  depends_on = [google_service_account.workbench_sa]
}

# Crea una instancia de Vertex AI Workbench
resource "google_workbench_instance" "instance" {
  name = "mit-project-workbench-instance"
  location = var.gcp_zone
  project = var.gcp_project_id

  gce_setup {
    machine_type = var.vm_machine_type
    service_accounts {
      email = google_service_account.workbench_sa.email
      }
  }
  
  labels = {
    environment = "dev"
    type = "workbench"
    app = "mit-project"
  }

  depends_on = [
    google_project_service.vertex_ai_api, 
    google_service_account.workbench_sa
    ]
}