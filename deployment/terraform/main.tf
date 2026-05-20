terraform {
  required_version = ">= 1.6.0"
}

variable "environment" {
  type    = string
  default = "staging"
}

output "aligngpt_environment" {
  value = var.environment
}
