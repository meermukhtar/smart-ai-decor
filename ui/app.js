// SmartSpace AI Frontend Client & 3D Studio
document.addEventListener("DOMContentLoaded", () => {
  // 1. Templates / Presets (Architectural Grade)
  const PRESETS = {
    bedroom_twin_study: {
      room_type: "bedroom",
      style: "modern_luxury",
      description: "Architectural luxury master bedroom suite: low-profile fluted walnut platform king bed centered against an acoustic wood-slat feature accent wall, layered neutral linen duvet and accent pillows, symmetrical floating nightstands with warm amber globe lamps. On the side zone, compact Scandinavian dual-crib nursery setup with organic cotton bedding and woven storage baskets. Dedicated study zone: sleek floating oak study desk with ergonomic Herman Miller style black mesh chair and 34-inch curved monitor. Floor-to-ceiling sheer linen curtains, tall potted fiddle leaf fig in ceramic planter, warm 2700K recessed cove ceiling lighting, textured wool area rug, Architectural Digest 8k UHD."
    },
    scandinavian_living: {
      room_type: "living_room",
      style: "scandinavian",
      description: "Editorial Scandinavian luxury living room: curved low-profile off-white bouclé sectional sofa anchored by an oversized textured geometric wool rug, fluted natural white oak and honed Carrara marble coffee table, minimalist floating media console with 65-inch ultra-thin TV, architectural matte black standing floor lamp, tall potted fiddle-leaf fig tree by sunlit bay windows, soft natural daylight with sheer drapes, warm minimalist aesthetic, ray-traced shadows, 8k UHD."
    },
    executive_office: {
      room_type: "office",
      style: "modern_luxury",
      description: "Prestigious modern executive home office: statement dark walnut executive desk with brushed brass cable grommets and minimalist desk pad, Herman Miller Eames style black leather executive chair, dual 32-inch bezel-less displays, floor-to-ceiling dark oak integrated library bookshelf with subtle LED shelf lighting, cozy bouclé reading armchair with brass side table in corner, dark herringbone hardwood floors, Architectural Digest photography, sharp focus 8k."
    },
    home_gym: {
      room_type: "gym",
      style: "industrial",
      description: "Sleek commercial-grade modern home gym studio: heavy-duty seamless black interlocking rubber shock-absorbing floor mats, compact matte-black steel dumbbell rack with chrome weights, adjustable leather incline workout bench, full-length illuminated LED mirror wall, wall-mounted matte pull-up bar, air-purifying cascading pothos plant, bright energized daylight studio lighting, clean industrial luxury aesthetic, 8k uhd."
    },
    minimal_dining: {
      room_type: "dining_room",
      style: "minimalist",
      description: "High-end minimalist architectural dining space: solid European white oak 8-person dining table with soft beveled edges, sculptural upholstered curved dining chairs, dramatic horizontal linear brass pendant light fixture centered above table, large monochromatic textured canvas wall art, floor-to-ceiling sheer drapery diffusing soft morning daylight, uncluttered spatial elegance, 8k uhd architectural photography."
    },
    cozy_lounge: {
      room_type: "living_room",
      style: "cozy_contemporary",
      description: "Inviting cozy contemporary lounge retreat: deep-seated low-profile modular cloud sofa in warm oatmeal fabric with plush cashmere throw blankets, nested travertine and smoked oak coffee tables, thick layered Moroccan Berber wool rug, sculptural plaster wall sconces emitting warm ambient 2400K glow, olive tree in terracotta pot, Architectural Digest interior, photorealistic 8k."
    },
    penthouse_suite: {
      room_type: "bedroom",
      style: "modern_luxury",
      description: "Ultra-luxury penthouse master suite: Italian leather upholstered king bed, custom bronze-framed headboard wall paneling, integrated floating bedside drawers, marble-topped credenza, velvet chaise lounge near expansive floor-to-ceiling glass windows, warm ambient architectural lighting, photorealistic 8k uhd."
    },
    japandi_zen: {
      room_type: "bedroom",
      style: "minimalist",
      description: "Minimalist Japandi zen master bedroom: low tatami-inspired platform bed with neutral linen duvet, natural bamboo accents, paper lantern pendant light, bonsai tree on floating wood shelf, clean uncluttered space, soft diffused sunlight, calm tranquil luxury, 8k uhd."
    }
  };

  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const imageInput = document.getElementById("imageInput");
  const previewContainer = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");
  const btnSample = document.getElementById("btnSample");
  const roomType = document.getElementById("roomType");
  const styleSelect = document.getElementById("styleSelect");
  const descriptionInput = document.getElementById("descriptionInput");
  const btnEnhancePrompt = document.getElementById("btnEnhancePrompt");
  const meterBadge = document.getElementById("meterBadge");
  const btnSubmit = document.getElementById("btnSubmit");
  const loadingBox = document.getElementById("loadingBox");
  const loadingText = document.getElementById("loadingText");
  const emptyPlaceholder = document.getElementById("emptyPlaceholder");
  const resultsStudio = document.getElementById("resultsStudio");

  // Studio Elements
  const stageImage = document.getElementById("stageImage");
  const currentAngleBadge = document.getElementById("currentAngleBadge");
  const anglePillsContainer = document.getElementById("anglePillsContainer");
  const angleSlider = document.getElementById("angleSlider");
  const btnToggleBeforeAfter = document.getElementById("btnToggleBeforeAfter");
  const btnDownloadAngle = document.getElementById("btnDownloadAngle");
  const btnDownloadOBJ = document.getElementById("btnDownloadOBJ");
  const btnDownloadPLY = document.getElementById("btnDownloadPLY");
  const btnCopyJson = document.getElementById("btnCopyJson");
  const jsonViewer = document.getElementById("jsonViewer");

  let selectedFile = null;
  let currentResponse = null;
  let currentAngleIndex = 0;
  let isShowingBefore = false;

  // Set default preset
  applyPreset("bedroom_twin_study");

  // Preset Buttons Click
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      applyPreset(btn.dataset.preset);
    });
  });

  function applyPreset(key) {
    const p = PRESETS[key];
    if (!p) return;
    roomType.value = p.room_type;
    styleSelect.value = p.style;
    descriptionInput.value = p.description;
    updatePromptScore();
  }

  // Quick-Add Feature Chips
  document.querySelectorAll(".prompt-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const tag = chip.dataset.chip;
      let cur = descriptionInput.value.trim();
      if (!cur) {
        descriptionInput.value = tag;
      } else if (!cur.toLowerCase().includes(tag.toLowerCase())) {
        descriptionInput.value = `${cur}, ${tag}`;
      }
      updatePromptScore();
      // Subtle pulse animation
      chip.style.transform = "scale(1.15)";
      setTimeout(() => { chip.style.transform = "scale(1)"; }, 180);
    });
  });

  // Real-Time Prompt Quality Meter
  function updatePromptScore() {
    if (!meterBadge) return;
    const text = (descriptionInput.value || "").toLowerCase();
    
    // Check key ingredients
    const materials = ["walnut", "bouclé", "marble", "brass", "wood", "linen", "oak", "leather", "rubber"];
    const lighting = ["warm", "cove", "ambient", "2700k", "sunlight", "pendant", "sconce", "lighting"];
    const photo = ["8k", "architectural", "photorealistic", "digest", "sharp focus"];
    
    let hasMaterial = materials.some(m => text.includes(m));
    let hasLighting = lighting.some(l => text.includes(l));
    let hasPhoto = photo.some(p => text.includes(p));

    if (text.length > 120 && hasMaterial && hasLighting) {
      meterBadge.className = "meter-badge badge-luxury";
      meterBadge.innerText = "⭐ High-End Architectural (Optimal Realism)";
    } else if (text.length > 50 && (hasMaterial || hasLighting)) {
      meterBadge.className = "meter-badge badge-good";
      meterBadge.innerText = "🔵 Good Spatial Details";
    } else {
      meterBadge.className = "meter-badge badge-basic";
      meterBadge.innerText = "🟡 Basic Prompt (Click ✨ AI Polish)";
    }
  }

  descriptionInput.addEventListener("input", updatePromptScore);

  // One-Click AI Prompt Polish & Enhance Button
  if (btnEnhancePrompt) {
    btnEnhancePrompt.addEventListener("click", () => {
      let cur = descriptionInput.value.trim();
      const rType = roomType.value;
      const sStyle = styleSelect.value;

      if (!cur) {
        cur = `Stage this ${rType.replace('_', ' ')} with high-end designer furniture`;
      }

      // Append elite lighting and designer finishes if missing
      const additions = [];
      const lower = cur.toLowerCase();

      if (!lower.includes("lighting") && !lower.includes("cove") && !lower.includes("sunlight")) {
        additions.push("warm 2700K recessed cove ceiling lighting, soft natural daylight from windows");
      }
      if (!lower.includes("rug") && !lower.includes("floor")) {
        additions.push("oversized textured plush wool area rug");
      }
      if (!lower.includes("fluted") && !lower.includes("bouclé") && !lower.includes("marble") && !lower.includes("walnut")) {
        additions.push("natural fluted white oak and honed Carrara marble finishes");
      }
      if (!lower.includes("plant") && !lower.includes("fig")) {
        additions.push("tall potted indoor fiddle-leaf fig in modern ceramic planter");
      }
      if (!lower.includes("8k") && !lower.includes("architectural digest")) {
        additions.push("Architectural Digest 8k UHD, photorealistic sharp focus, uncluttered spatial balance");
      }

      if (additions.length > 0) {
        descriptionInput.value = `${cur}. Integrated styling: ${additions.join(", ")}.`;
      }

      updatePromptScore();
      btnEnhancePrompt.innerText = "✓ Enhanced!";
      setTimeout(() => { btnEnhancePrompt.innerText = "✨ AI Polish & Enhance"; }, 2000);
    });
  }

  // File Upload & Dropzone
  dropzone.addEventListener("click", () => imageInput.click());
  
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  imageInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewContainer.style.display = "block";
    };
    reader.readAsDataURL(file);
  }

  // Load Default Sample Room
  btnSample.addEventListener("click", async () => {
    try {
      btnSample.innerText = "⏳ Loading sample...";
      // Fetch sample room from server or fallback
      const resp = await fetch("/media/rooms/rooms.jpeg").catch(() => null);
      if (resp && resp.ok) {
        const blob = await resp.blob();
        const file = new File([blob], "sample_room.jpeg", { type: "image/jpeg" });
        handleFile(file);
        btnSample.innerText = "✓ Sample Loaded (rooms.jpeg)";
      } else {
        // Fetch any existing room from media
        const altResp = await fetch("/media/rooms/rooms_Y8AuwFJ.jpeg");
        const blob = await altResp.blob();
        const file = new File([blob], "sample_room.jpeg", { type: "image/jpeg" });
        handleFile(file);
        btnSample.innerText = "✓ Sample Loaded";
      }
    } catch (err) {
      console.warn("Sample load notice:", err);
      btnSample.innerText = "⚡ Use Sample Empty Room";
      alert("Please choose an empty room image from your computer.");
      imageInput.click();
    }
  });

  // Tab Switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) {
        target.classList.add("active");
        if (btn.dataset.tab === "tab-3d") {
          onTab3DSelected();
        }
      }
    });
  });

  // Submit / Stage Room
  btnSubmit.addEventListener("click", async () => {
    if (!selectedFile) {
      alert("Please upload or select an empty room image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("room_type", roomType.value);
    formData.append("style", styleSelect.value);
    formData.append("description", descriptionInput.value);

    // Show Progress
    btnSubmit.disabled = true;
    loadingBox.style.display = "block";

    const messages = [
      "Analyzing empty room architecture & lighting...",
      "Segmenting floor footprint & detecting bounds...",
      "Running Local RTX 3080 Ti Diffusion Staging...",
      "Generating 3D surface geometry & normals...",
      "Synthesizing 5 multi-angle camera perspectives...",
      "Finalizing high-definition rendering..."
    ];
    let msgIdx = 0;
    loadingText.innerText = messages[0];
    const timer = setInterval(() => {
      msgIdx = (msgIdx + 1) % messages.length;
      loadingText.innerText = messages[msgIdx];
    }, 4500);

    try {
      const apiHost = window.location.port === "8000" ? "" : "http://127.0.0.1:8000";
      const res = await fetch(`${apiHost}/api/rooms/auto-decorate/`, {
        method: "POST",
        body: formData,
      });

      clearInterval(timer);
      loadingBox.style.display = "none";
      btnSubmit.disabled = false;

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      currentResponse = data;
      renderResults(data);

    } catch (err) {
      clearInterval(timer);
      loadingBox.style.display = "none";
      btnSubmit.disabled = false;
      alert(`Staging failed: ${err.message}`);
    }
  });

  // Render Results
  function renderResults(data) {
    emptyPlaceholder.style.display = "none";
    resultsStudio.style.display = "flex";

    // Summary Header
    document.getElementById("resRoomTitle").innerText = `${data.room_name || 'Room'} (${data.applied_style || 'modern_luxury'})`;
    document.getElementById("badgeRoomType").innerText = data.room_type || 'bedroom';
    document.getElementById("badgeStyle").innerText = data.applied_style || 'modern_luxury';
    
    const dims = data.free_space_summary || {};
    document.getElementById("resDimensions").innerText = `Est. Free Floor: ${dims.free_floor_area_sqm || 4.0} m² | ${dims.free_space_percentage || 100}% Staged`;

    // 1. Setup Multi-Angle Rotator
    setupMultiAngleViews(data.multi_angle_views || [], data.original_image_url);

    // 2. Setup Spatial Analysis
    document.getElementById("spatialAnnotatedImg").src = data.annotated_image_url || "";
    document.getElementById("spatialDepthImg").src = data.depth_map_url || "";

    // Placed items table
    const tableBody = document.getElementById("placedItemsTableBody");
    tableBody.innerHTML = "";
    (data.placed_items || []).forEach(item => {
      const tr = document.createElement("tr");
      const pos = item.position || {};
      tr.innerHTML = `
        <td style="font-weight: 600; color: #f8fafc;">${item.title || item.item_name}</td>
        <td><span style="background: rgba(255,255,255,0.06); padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">${item.category}</span></td>
        <td style="font-family: monospace; color: #94a3b8;">(${pos.x}, ${pos.y}) [${pos.width}x${pos.height}]</td>
      `;
      tableBody.appendChild(tr);
    });

    // Suggestions list
    const suggList = document.getElementById("suggestionsList");
    suggList.innerHTML = "";
    (data.suggestions || []).forEach(s => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${s.title}</strong>: ${s.reason}`;
      suggList.appendChild(li);
    });

    // 3. Raw JSON & Downloads
    jsonViewer.innerText = JSON.stringify(data, null, 2);
    if (data.mesh_3d_url) {
      btnDownloadOBJ.href = data.mesh_3d_url;
      btnDownloadOBJ.style.display = "inline-block";
    } else {
      btnDownloadOBJ.style.display = "none";
    }

    if (data.point_cloud_url) {
      btnDownloadPLY.href = data.point_cloud_url;
      btnDownloadPLY.style.display = "inline-block";
    } else {
      btnDownloadPLY.style.display = "none";
    }

    // 4. Initialize 3D Mesh
    init3DStudio(data.mesh_3d_url, data.point_cloud_url);
  }

  // Multi-Angle Views Handling
  function setupMultiAngleViews(views, originalUrl) {
    anglePillsContainer.innerHTML = "";
    currentAngleIndex = 0;
    isShowingBefore = false;

    if (!views || views.length === 0) {
      stageImage.src = currentResponse.decorated_image_url || "";
      currentAngleBadge.innerText = "Front View (0°)";
      return;
    }

    angleSlider.max = views.length - 1;
    angleSlider.value = 0;

    views.forEach((v, idx) => {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = `angle-pill ${idx === 0 ? 'active' : ''}`;
      pill.innerText = v.label || v.angle;
      pill.addEventListener("click", () => selectAngle(idx));
      anglePillsContainer.appendChild(pill);
    });

    selectAngle(0);

    // Slider change
    angleSlider.oninput = (e) => {
      selectAngle(parseInt(e.target.value, 10));
    };

    // Toggle Before / After
    btnToggleBeforeAfter.onclick = () => {
      isShowingBefore = !isShowingBefore;
      if (isShowingBefore) {
        stageImage.src = originalUrl || "";
        currentAngleBadge.innerText = "Empty Original Room";
        btnToggleBeforeAfter.innerText = "Show Staged Design";
      } else {
        selectAngle(currentAngleIndex);
        btnToggleBeforeAfter.innerText = "Toggle Before / After";
      }
    };
  }

  function selectAngle(index) {
    if (!currentResponse || !currentResponse.multi_angle_views) return;
    const views = currentResponse.multi_angle_views;
    if (index < 0 || index >= views.length) return;

    currentAngleIndex = index;
    isShowingBefore = false;
    btnToggleBeforeAfter.innerText = "Toggle Before / After";

    const view = views[index];
    stageImage.src = view.image_url;
    currentAngleBadge.innerText = view.label || view.angle;
    btnDownloadAngle.href = view.image_url;
    angleSlider.value = index;

    // Update active pill
    const pills = anglePillsContainer.querySelectorAll(".angle-pill");
    pills.forEach((p, idx) => {
      if (idx === index) p.classList.add("active");
      else p.classList.remove("active");
    });
  }

  // Copy JSON Button
  btnCopyJson.addEventListener("click", () => {
    if (!currentResponse) return;
    navigator.clipboard.writeText(JSON.stringify(currentResponse, null, 2));
    btnCopyJson.innerText = "✓ Copied!";
    setTimeout(() => { btnCopyJson.innerText = "📋 Copy JSON"; }, 2000);
  });

  // ==========================================
  // Three.js Interactive 3D Studio
  // ==========================================
  let scene, camera, renderer, controls, currentMeshGroup;
  let isAutoSpin = true;

  function init3DStudio(meshUrl, plyUrl) {
    const container = document.getElementById("canvas-3d");
    if (!container) return;

    if (!scene) {
      // Scene
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x070a0f);

      // Camera
      camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 100);
      camera.position.set(0, 1.2, 3.5);

      // Renderer
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputEncoding = THREE.sRGBEncoding;
      container.innerHTML = "";
      container.appendChild(renderer.domElement);

      // Controls
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.autoRotate = isAutoSpin;
      controls.autoRotateSpeed = 1.2;
      controls.target.set(0, 0, -1.8);

      // Lighting
      const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.7);
      hemiLight.position.set(0, 20, 0);
      scene.add(hemiLight);

      const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
      dirLight1.position.set(5, 10, 7);
      scene.add(dirLight1);

      const dirLight2 = new THREE.DirectionalLight(0xa5b4fc, 0.5);
      dirLight2.position.set(-5, -5, -5);
      scene.add(dirLight2);

      // Actions
      document.getElementById("btnResetCamera").onclick = () => {
        camera.position.set(0, 1.2, 3.5);
        controls.target.set(0, 0, -1.8);
        controls.update();
      };

      document.getElementById("btnToggleAutoSpin").onclick = () => {
        isAutoSpin = !isAutoSpin;
        controls.autoRotate = isAutoSpin;
        document.getElementById("btnToggleAutoSpin").innerText = isAutoSpin ? "Pause Spin" : "Auto-Rotate";
      };

      // Resize observer
      window.addEventListener("resize", () => {
        if (!container.clientWidth || !container.clientHeight) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
      });

      // Animation loop
      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }
      animate();
    }

    // Load Model
    loadRoomModel(meshUrl, plyUrl);
  }

  function loadRoomModel(meshUrl, plyUrl) {
    if (currentMeshGroup) {
      scene.remove(currentMeshGroup);
      currentMeshGroup = null;
    }

    if (!meshUrl && !plyUrl) return;

    currentMeshGroup = new THREE.Group();
    scene.add(currentMeshGroup);

    // Try OBJLoader first
    if (meshUrl && typeof THREE.OBJLoader !== "undefined") {
      const loader = new THREE.OBJLoader();
      loader.load(
        meshUrl,
        (obj) => {
          obj.traverse((child) => {
            if (child.isMesh) {
              child.material = new THREE.MeshStandardMaterial({
                vertexColors: true,
                roughness: 0.55,
                metalness: 0.15,
                side: THREE.DoubleSide
              });
            }
          });
          currentMeshGroup.add(obj);
        },
        undefined,
        (err) => {
          console.warn("OBJ load fallback to PLY:", err);
          loadPLYFallback(plyUrl);
        }
      );
    } else if (plyUrl) {
      loadPLYFallback(plyUrl);
    }
  }

  function loadPLYFallback(plyUrl) {
    if (!plyUrl || typeof THREE.PLYLoader === "undefined") return;
    const loader = new THREE.PLYLoader();
    loader.load(plyUrl, (geometry) => {
      geometry.computeVertexNormals();
      const material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.5,
        metalness: 0.1,
        side: THREE.DoubleSide
      });
      const mesh = new THREE.Mesh(geometry, material);
      if (currentMeshGroup) currentMeshGroup.add(mesh);
    });
  }

  function onTab3DSelected() {
    setTimeout(() => {
      const container = document.getElementById("canvas-3d");
      if (renderer && camera && container) {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
      }
    }, 50);
  }
});
