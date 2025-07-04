import os
import re
import signal
import subprocess
import atexit
import json
from flask import Flask, jsonify, render_template_string

# --- Configuration ---
PROJECTS_DIRECTORY = "/nvme/scratch/work/tharvey/fitsmap/"
STARTING_PORT = 4021  # Starting port for serving projects

app = Flask(__name__)
processes = {}


def cleanup_processes():
    """Ensure all child processes are killed on exit via SIGINT."""
    print("Shutting down... sending interrupt to all fitsmap services.")
    for project_name in list(processes.keys()):
        process_info = processes.pop(project_name)
        try:
            os.killpg(os.getpgid(process_info["process"].pid), signal.SIGINT)
            print(f"Stopped {project_name}.")
        except ProcessLookupError:
            pass


atexit.register(cleanup_processes)


def scan_projects_recursive(base_path: str, current_path: str) -> list[dict]:
    """
    Recursively scans a directory to build a nested structure of folders and projects.
    """
    items = []
    for item_name in sorted(os.listdir(current_path)):
        full_path = os.path.join(current_path, item_name)

        if not os.path.isdir(full_path):
            continue

        # Check if the directory is a fitsmap project itself
        is_project = (
            "index.html" in os.listdir(full_path)
            and "js" in os.listdir(full_path)
            and "css" in os.listdir(full_path)
        )

        if is_project:
            # It's a project, add it to the list
            relative_path = os.path.relpath(full_path, base_path)
            match = re.match(r"^(.*?)_v(\d+)$", item_name)
            if match:
                base_name, version = match.groups()
                items.append(
                    {
                        "type": "project",
                        "full_name": relative_path,
                        "base_name": base_name.replace("_", " "),
                        "version": f"v{version}",
                    }
                )
            else:
                items.append(
                    {
                        "type": "project",
                        "full_name": relative_path,
                        "base_name": item_name.replace("_", " "),
                        "version": None,
                    }
                )
        else:
            # It's a folder, scan its children recursively
            children = scan_projects_recursive(base_path, full_path)
            if children:  # Only add folders that contain projects
                items.append({"type": "folder", "name": item_name, "children": children})
    return items


@app.route("/")
def index():
    """Renders the homepage template."""
    if not os.path.exists(PROJECTS_DIRECTORY):
        print(f"FATAL: Projects directory not found at '{PROJECTS_DIRECTORY}'")
        projects = []
    else:
        projects = scan_projects_recursive(PROJECTS_DIRECTORY, PROJECTS_DIRECTORY)
    return render_template_string(HOME_PAGE_TEMPLATE, projects=projects)


@app.route("/launch/<path:project_name>")
def launch_project(project_name: str):
    """Launches a fitsmap project if not already running."""
    if project_name in processes and processes[project_name]["process"].poll() is None:
        port = processes[project_name]["port"]
        return jsonify({"url": f"http://127.0.0.1:{port}", "restarted": False})

    port = STARTING_PORT + len(processes)
    # Project name is now a relative path, join it with the base directory
    project_path = os.path.join(PROJECTS_DIRECTORY, project_name)

    if not os.path.isdir(project_path):
        return jsonify({"error": f"Project not found at path: {project_path}"}), 404

    process = subprocess.Popen(
        ["fitsmap", "serve", "--port", str(port), "--open_browser", "false"],
        cwd=project_path,
        preexec_fn=os.setsid,
    )
    processes[project_name] = {"process": process, "port": port}
    return jsonify({"url": f"http://127.0.0.1:{port}", "restarted": True})


@app.route("/info/<path:project_name>")
def get_project_info(project_name: str):
    """Returns the content of the info.json file for a project."""
    info_path = os.path.join(PROJECTS_DIRECTORY, project_name, "info.json")
    print(info_path)
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            try:
                info_data = json.load(f)
                print(info_data)

                return jsonify(info_data)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format"}), 500
    return jsonify({"error": "info.json not found"}), 404


@app.route("/stop/<path:project_name>")
def stop_project(project_name: str):
    """Stops a fitsmap project by sending a SIGINT (Ctrl+C) signal."""
    if project_name in processes:
        process_info = processes.pop(project_name)
        try:
            os.killpg(os.getpgid(process_info["process"].pid), signal.SIGINT)
            return jsonify({"status": "stopped"})
        except ProcessLookupError:
            return jsonify({"status": "already stopped"})
    return jsonify({"error": "Project not running"}), 404


# --- HTML, CSS, and JavaScript Template ---
HOME_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FitsMap Launcher</title>
    <style>
        :root { --sidebar-width: 320px; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; display: flex; height: 100vh; margin: 0; background-color: #f8f9fa; }
        #sidebar { width: var(--sidebar-width); background-color: #343a40; color: white; padding: 20px; display: flex; flex-direction: column; flex-shrink: 0; border-right: 1px solid #495057; overflow-y: auto; position: relative; }
        #sidebar h1 { font-size: 1.6em; margin-bottom: 25px; font-weight: 500; }
        #project-list, #project-list ul { list-style: none; padding: 0; margin: 0; }
        #project-list ul { padding-left: 20px; }
        #project-list a { color: #dee2e6; text-decoration: none; display: flex; justify-content: space-between; align-items: center; padding: 10px 15px; border-radius: 5px; transition: background-color 0.2s, color 0.2s; }
        #project-list a:hover, #project-list a.active { background-color: #007bff; color: white; }
        details { margin-bottom: 5px; }
        summary { cursor: pointer; padding: 10px 15px; border-radius: 5px; font-weight: 600; list-style: '📂 '; }
        details[open] > summary { list-style: '📁 '; }
        summary:hover { background-color: #495057; }
        #info-box { position: absolute; bottom: 20px; right: 20px; width: calc(var(--sidebar-width) - 40px); background-color: #212529; border: 1px solid #495057; border-radius: 8px; padding: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); display: none; z-index: 1000; }
        #info-box h3 { margin-top: 0; margin-bottom: 10px; color: #00aaff; }
        #info-box p { margin: 0 0 5px; font-size: 0.9em; }
        #info-box strong { color: #adb5bd; }
        #content { flex-grow: 1; display: flex; flex-direction: column; }
        #topbar { display: flex; justify-content: flex-end; padding: 10px; background-color: #ffffff; border-bottom: 1px solid #dee2e6; }
        #topbar button { display: none; margin-left: 10px; padding: 8px 15px; border: 1px solid #6c757d; border-radius: 5px; cursor: pointer; background-color: white; font-weight: 500; }
        #topbar #fullscreen-btn { background-color: #007bff; color: white; border-color: #007bff; }
        #topbar #stop-btn { background-color: #dc3545; color: white; border-color: #dc3545; }
        #iframe-container { flex-grow: 1; position: relative; background-color: #f1f3f5; }
        iframe { width: 100%; height: 100%; border: none; }
        .loader { text-align: center; padding-top: 50px; font-size: 1.2em; color: #6c757d; }
        .version-badge { background-color: #28a745; color: white; font-size: 0.8em; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>FitsMap Projects</h1>
        <ul id="project-list">
            {% macro render_items(items) %}
                {% for item in items %}
                    <li>
                        {% if item.type == 'folder' %}
                            <details>
                                <summary>{{ item.name }}</summary>
                                <ul>{{ render_items(item.children) }}</ul>
                            </details>
                        {% elif item.type == 'project' %}
                            <a href="#" id="link-{{ item.full_name | replace('/', '-') }}" 
                               data-project-name="{{ item.full_name }}" 
                               data-has-info="{{ 'true' if item.has_info else 'false' }}"
                               onclick="event.preventDefault(); launch(this)"
                               onmouseover="showInfo(this)" onmouseout="hideInfo()">
                                <span>{{ item.base_name }}</span>
                                {% if item.version %}
                                    <span class="version-badge">{{ item.version }}</span>
                                {% endif %}
                            </a>
                        {% endif %}
                    </li>
                {% endfor %}
            {% endmacro %}
            {{ render_items(projects) }}
        </ul>
        <div id="info-box"></div>
    </div>
    <div id="content">
        <div id="topbar">
            <button id="fullscreen-btn" onclick="toggleFullScreen()">Full Screen</button>
            <button id="stop-btn" onclick="stopProject()">Stop</button>
        </div>
        <div id="iframe-container">
            <iframe id="project-frame" name="project-frame"></iframe>
        </div>
    </div>

    <script>
        let currentProject = null;
        const iframe = document.getElementById('project-frame');
        const fullscreenBtn = document.getElementById('fullscreen-btn');
        const stopBtn = document.getElementById('stop-btn');
        const infoBox = document.getElementById('info-box');

        function showLoader() {
            iframe.style.display = 'none';
            const existingLoader = document.querySelector('.loader');
            if (existingLoader) existingLoader.remove();
            const loaderDiv = document.createElement('div');
            loaderDiv.className = 'loader';
            loaderDiv.textContent = 'Launching project, please wait... 🚀';
            document.getElementById('iframe-container').appendChild(loaderDiv);
        }

        function hideLoader() {
            const loader = document.querySelector('.loader');
            if (loader) loader.remove();
            iframe.style.display = 'block';
        }

        async function launch(linkElement) {
            const projectName = linkElement.getAttribute('data-project-name');
            if (currentProject === projectName) return;

            showLoader();
            if (currentProject) {
                const oldLink = document.getElementById(`link-${currentProject.replace(/\\//g, '-')}`);
                if (oldLink) oldLink.classList.remove('active');
            }
            
            const response = await fetch(`/launch/${encodeURIComponent(projectName)}`);
            const data = await response.json();

            if (data.url) {
                currentProject = projectName;
                linkElement.classList.add('active');
                
                const startupDelay = data.restarted ? 5000 : 0;
                setTimeout(() => {
                    iframe.src = data.url;
                    hideLoader();
                }, startupDelay);

                fullscreenBtn.style.display = 'inline-block';
                stopBtn.style.display = 'inline-block';
            } else {
                alert('Error launching project: ' + (data.error || 'Unknown error'));
                hideLoader();
            }
        }

        async function stopProject() {
            if (currentProject) {
                await fetch(`/stop/${encodeURIComponent(currentProject)}`);
                iframe.src = 'about:blank';
                const currentLink = document.getElementById(`link-${currentProject.replace(/\\//g, '-')}`);
                if (currentLink) currentLink.classList.remove('active');
                currentProject = null;
                fullscreenBtn.style.display = 'none';
                stopBtn.style.display = 'none';
            }
        }
        
        async function showInfo(linkElement) {
            if (linkElement.getAttribute('data-has-info') !== 'true') return;
            const projectName = linkElement.getAttribute('data-project-name');
            const response = await fetch(`/info/${encodeURIComponent(projectName)}`);
            const data = await response.json();
            
            if (data.error) {
                infoBox.innerHTML = `<p>Error: ${data.error}</p>`;
            } else {
                let content = `<h3>${projectName.split('/').pop()} Info</h3>`;
                for (const [key, value] of Object.entries(data)) {
                    content += `<p><strong>${key}:</strong> ${value}</p>`;
                }
                infoBox.innerHTML = content;
            }
            infoBox.style.display = 'block';
        }

        function hideInfo() {
            infoBox.style.display = 'none';
        }

        function toggleFullScreen() {
            const el = document.getElementById('iframe-container');
            if (el.requestFullscreen) el.requestFullscreen();
            else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
            else if (el.msRequestFullscreen) el.msRequestFullscreen();
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    # app.run(debug=True, port=STARTING_PORT, use_reloader=False)

    app.run(host="0.0.0.0", port=STARTING_PORT, use_reloader=False)
