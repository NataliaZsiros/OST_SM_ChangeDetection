# Stage 1: Base Java environment
FROM eclipse-temurin:11-jdk as java_base
 
# Set JAVA_HOME environment variable
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"
 
# Stage 2: Base Python environment
FROM python:3.9.6 as local_base
 
# Copy Java runtime from the previous stage
COPY --from=java_base /opt/java /opt/java
 
# Set JAVA_HOME environment variable in the Python image
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"
 
# Set working directory
WORKDIR /dependencies
 
# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt