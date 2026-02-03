# Local MediaWiki with Semantic MediaWiki and extensions
# Based on MediaWiki 1.43 with additional extensions for knowledge management
# Note: Upgraded from 1.41.2 to support Network extension for graph visualization

FROM mediawiki:1.43

# Install system dependencies:
# - git, unzip, wget: for downloading extensions
# - default-mysql-client: for database health checks
# - python3, python3-pygments: for SyntaxHighlight extension
# - nodejs, npm: for Mermaid diagrams
RUN apt-get update \
 && apt-get install -y git unzip wget ca-certificates default-mysql-client \
    python3 python3-pygments nodejs npm \
 && rm -rf /var/lib/apt/lists/*

# Install Composer for PHP dependencies
RUN wget -O /usr/local/bin/composer https://getcomposer.org/download/latest-stable/composer.phar \
 && chmod +x /usr/local/bin/composer

WORKDIR /var/www/html

# Install Semantic MediaWiki and Mermaid via Composer
# SMW 5.x is required for MediaWiki 1.43 compatibility (DB_MASTER → DB_PRIMARY)
# Disable audit blocking for phpunit (dev dependency with security advisory)
ENV COMPOSER_NO_AUDIT=1
RUN set -eux; \
    composer config --global audit.block-insecure false; \
    printf '{ "require": { "mediawiki/semantic-media-wiki": "^5.0", "mediawiki/mermaid": "^3.0" }, "extra": { "merge-plugin": { "include": ["extensions/SemanticMediaWiki/composer.json", "extensions/Mermaid/composer.json"] } } }\n' > composer.local.json; \
    composer update --no-dev --no-audit

# Install Mermaid JavaScript dependencies for diagram rendering
RUN set -eux; \
    if [ -d "extensions/Mermaid" ]; then \
        cd extensions/Mermaid && npm install && cd ../..; \
    fi

# Clone additional extensions from GitHub
# All extensions use the REL1_43 branch for compatibility
ENV MW_EXT_REF=REL1_43
RUN set -eux; \
    fetch() { \
      name="$1"; url="$2"; \
      if [ ! -f "extensions/${name}/extension.json" ]; then \
        echo "Cloning ${name}..."; \
        git clone --depth 1 --branch "$MW_EXT_REF" "$url" "extensions/${name}"; \
      else \
        echo "Skipping ${name} (already present)"; \
      fi; \
    }; \
    fetch PageForms     https://github.com/wikimedia/mediawiki-extensions-PageForms.git; \
    fetch Cargo         https://github.com/wikimedia/mediawiki-extensions-Cargo.git; \
    fetch VisualEditor  https://github.com/wikimedia/mediawiki-extensions-VisualEditor.git; \
    fetch SyntaxHighlight_GeSHi https://github.com/wikimedia/mediawiki-extensions-SyntaxHighlight_GeSHi.git; \
    fetch Math          https://github.com/wikimedia/mediawiki-extensions-Math.git

# DynamicPageList3 - skip for now as main branch (DPL4) requires MW 1.44+
# TODO: Find a compatible version or upgrade MediaWiki
# RUN git clone --depth 1 --branch main https://github.com/Universal-Omega/DynamicPageList3.git extensions/DynamicPageList3

# Clone Network extension for interactive page graph visualization
# This extension uses main branch (not REL1_43)
RUN git clone --depth 1 https://github.com/ProfessionalWiki/Network.git extensions/Network \
    && cd extensions/Network && composer install --no-dev --no-interaction && cd ../..

# Copy entrypoint script and helper files
COPY entrypoint.sh /entrypoint.sh
COPY import_wikis.sh /import_wikis.sh
COPY create_user.php /create_user.php
COPY leeroo-logo.png /var/www/html/leeroo-logo.png
RUN chmod +x /entrypoint.sh /import_wikis.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["apache2-foreground"]

