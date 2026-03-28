"""
Honeypot URL patterns.

These map common scanner targets to trap handlers.
Include in root urls.py — they sit outside the real app's URL space.
"""

from django.urls import path

from . import honeypots

urlpatterns = [
    # WordPress
    path("wp-admin/", honeypots.wp_admin),
    path("wp-login.php", honeypots.wp_admin),
    path("wp-admin/install.php", honeypots.wp_admin),
    path("wordpress/wp-admin/", honeypots.wp_admin),
    path("blog/wp-admin/", honeypots.wp_admin),
    # phpMyAdmin
    path("phpmyadmin/", honeypots.phpmyadmin),
    path("pma/", honeypots.phpmyadmin),
    path("phpMyAdmin/", honeypots.phpmyadmin),
    path("mysql/", honeypots.phpmyadmin),
    path("dbadmin/", honeypots.phpmyadmin),
    path("myadmin/", honeypots.phpmyadmin),
    # Environment / secrets
    path(".env", honeypots.fake_env),
    path(".env.bak", honeypots.fake_env),
    path(".env.old", honeypots.fake_env),
    path(".env.production", honeypots.fake_env),
    path(".env.local", honeypots.fake_env),
    # Git
    path(".git/config", honeypots.fake_git),
    path(".git/HEAD", honeypots.fake_git),
    path(".gitignore", honeypots.fake_git),
    # Admin APIs
    path("api/v1/admin/users/", honeypots.fake_admin_api),
    path("api/v1/users/", honeypots.fake_admin_api),
    path("api/admin/", honeypots.fake_admin_api),
    # Backups
    path("backup.sql", honeypots.fake_backup),
    path("db.sql", honeypots.fake_backup),
    path("dump.sql", honeypots.fake_backup),
    path("database.sql", honeypots.fake_backup),
    path("backup.tar.gz", honeypots.fake_backup),
]
