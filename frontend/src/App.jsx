"use client"

import { useEffect, useMemo, useState, useRef } from "react"
import * as THREE from "three"

// StarfieldBackground Component
const StarfieldBackground = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 bg-gradient-to-b from-black via-blue-900 to-black opacity-50"></div>
      {[...Array(200)].map((_, i) => (
        <div
          key={i}
          className="absolute w-1 h-1 bg-white rounded-full opacity-70 animate-pulse"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 3}s`,
            animationDuration: `${2 + Math.random() * 3}s`,
          }}
        />
      ))}
    </div>
  )
}

// Controls Component
const Controls = ({ onSimulate, isLoading }) => {
  return (
    <div className="flex justify-center space-x-4 relative z-10">
      <button
        onClick={onSimulate}
        disabled={isLoading}
        className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors cursor-pointer"
      >
        {isLoading ? "Simulating..." : "🚀 Simulate Mission"}
      </button>
    </div>
  )
}

// Logs Component
const Logs = ({ log }) => {
  if (!log || log.length === 0) {
    return <div className="text-gray-400 text-sm">No logs available</div>
  }

  return (
    <div className="space-y-1">
      {log.map((entry, i) => (
        <div key={i} className="text-xs text-gray-300">
          {entry}
        </div>
      ))}
    </div>
  )
}

// Mission Details Panel Component
const MissionDetailsPanel = ({ sim }) => {
  if (!sim) return null

  const formatDateTime = (isoString) => {
    if (!isoString) return "N/A"
    try {
      const date = new Date(isoString)
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short'
      })
    } catch {
      return isoString
    }
  }

  const calculateFuelConsumption = () => {
    if (!sim?.rl?.path) return "N/A"
    // Rough estimation: 1 unit per 10km distance
    const totalDistance = sim.rl.path.reduce((sum, point, i) => {
      if (i === 0) return 0
      const prev = sim.rl.path[i - 1]
      // Calculate 3D distance if available, otherwise 2D
      const dx = point[0] - prev[0]
      const dy = point[1] - prev[1]
      const dz = (point.length === 3 && prev.length === 3) ? point[2] - prev[2] : 0
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
      return sum + distance
    }, 0)
    return `${Math.round(totalDistance / 1000)} kg`
  }

  const getHazardCount = () => {
    const hazards = sim?.hazards?.length || 0
    const obstacles = (sim?.rl?.obstaclesHazards?.length || 0) + (sim?.rl?.obstaclesDebris?.length || 0)
    return { hazards, obstacles }
  }

  const hazardCount = getHazardCount()

  return (
    <div className="absolute top-4 left-4 bg-black/80 backdrop-blur-sm rounded-lg p-4 border border-gray-600 text-white text-sm max-w-xs">
      <h4 className="text-lg font-semibold mb-3 text-blue-400">Mission Details</h4>
      
      <div className="space-y-2">
        <div className="flex justify-between">
          <span className="text-gray-300">Timeframe:</span>
          <span className="text-white font-mono text-xs">
            {sim?.timeframe ? 
              `${formatDateTime(sim.timeframe.start)} - ${formatDateTime(sim.timeframe.end)}` : 
              "Random Window"
            }
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-300">Fuel Est.:</span>
          <span className="text-yellow-400">{calculateFuelConsumption()}</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-300">Hazards:</span>
          <span className="text-red-400">{hazardCount.hazards}</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-300">Obstacles:</span>
          <span className="text-orange-400">{hazardCount.obstacles}</span>
        </div>
        
        {sim?.rl?.constraints && (
          <>
            <div className="flex justify-between">
              <span className="text-gray-300">Kp Max:</span>
              <span className={`${sim.rl.constraints.kpMax >= 7 ? 'text-red-400' : 'text-green-400'}`}>
                {sim.rl.constraints.kpMax}
              </span>
            </div>
            
            {sim.rl.constraints.suggestedDelayHours > 0 && (
              <div className="flex justify-between">
                <span className="text-gray-300">Delay:</span>
                <span className="text-yellow-400">{sim.rl.constraints.suggestedDelayHours}h</span>
              </div>
            )}
          </>
        )}
        
        {sim?.rl?.debug && (
          <div className="flex justify-between">
            <span className="text-gray-300">Debris TLEs:</span>
            <span className="text-cyan-400">{sim.rl.debug.tleCount || 0}</span>
          </div>
        )}
      </div>
    </div>
  )
}

// Three.js Scene Component with Enhanced Controls
const ThreeJSVisualization = ({ sim, animIndex }) => {
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const cameraRef = useRef(null)
  const spacecraftRef = useRef(null)
  const trajectoryLinesRef = useRef([])
  const animationRef = useRef(null)
  const controlsRef = useRef({
    mouseDown: false,
    previousMousePosition: { x: 0, y: 0 },
    cameraRotation: { x: 0, y: 0 },
    cameraDistance: 80000,
    cameraTarget: { x: 0, y: 0, z: 0 },
  })

  useEffect(() => {
    if (!mountRef.current) return

    // Scene setup
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000011)
    sceneRef.current = scene

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      2000000,
    )
    camera.position.set(0, 0, 80000)
    cameraRef.current = camera

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.setClearColor(0x000011)
    mountRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6)
    scene.add(ambientLight)

    const sunLight = new THREE.DirectionalLight(0xffffff, 1.5)
    sunLight.position.set(200000, 100000, 100000)
    sunLight.castShadow = true
    sunLight.shadow.mapSize.width = 2048
    sunLight.shadow.mapSize.height = 2048
    scene.add(sunLight)

    // Earth
    const earthGeometry = new THREE.SphereGeometry(6371, 64, 64)
    const earthMaterial = new THREE.MeshPhongMaterial({
      color: 0x1e90ff,
      shininess: 100,
      transparent: false,
      opacity: 1.0,
    })

    const earth = new THREE.Mesh(earthGeometry, earthMaterial)
    earth.position.set(0, 0, 0)
    earth.castShadow = true
    earth.receiveShadow = true
    scene.add(earth)

    // Earth atmosphere glow
    const earthGlowGeometry = new THREE.SphereGeometry(6800, 32, 32)
    const earthGlowMaterial = new THREE.MeshBasicMaterial({
      color: 0x87ceeb,
      transparent: true,
      opacity: 0.2,
      side: THREE.BackSide,
    })
    const earthGlow = new THREE.Mesh(earthGlowGeometry, earthGlowMaterial)
    earth.add(earthGlow)

    // Add continents simulation
    const continentGeometry = new THREE.SphereGeometry(6372, 32, 32)
    const continentMaterial = new THREE.MeshBasicMaterial({
      color: 0x228b22,
      transparent: true,
      opacity: 0.4,
    })
    const continents = new THREE.Mesh(continentGeometry, continentMaterial)
    earth.add(continents)

    // Moon with 3D positioning
    const moonGeometry = new THREE.SphereGeometry(1737, 32, 32)
    const moonMaterial = new THREE.MeshPhongMaterial({
      color: 0xe6e6e6,
      shininess: 10,
      transparent: false,
    })
    const moon = new THREE.Mesh(moonGeometry, moonMaterial)
    // Position Moon in 3D space - further out and at a slight angle
    moon.position.set(60000, 30000, 10000)
    moon.castShadow = true
    moon.receiveShadow = true
    scene.add(moon)

    // Moon craters
    const moonCratersGeometry = new THREE.SphereGeometry(1738, 16, 16)
    const moonCratersMaterial = new THREE.MeshBasicMaterial({
      color: 0xc0c0c0,
      transparent: true,
      opacity: 0.3,
    })
    const moonCraters = new THREE.Mesh(moonCratersGeometry, moonCratersMaterial)
    moon.add(moonCraters)

    // Spacecraft with 3D positioning
    const spacecraftGeometry = new THREE.ConeGeometry(1200, 4000, 8)
    const spacecraftMaterial = new THREE.MeshPhongMaterial({
      color: 0xffd700,
      shininess: 100,
      emissive: 0x444400,
    })
    const spacecraft = new THREE.Mesh(spacecraftGeometry, spacecraftMaterial)
    spacecraft.castShadow = true
    // Start spacecraft at Earth surface
    spacecraft.position.set(6371, 0, 0)
    scene.add(spacecraft)
    spacecraftRef.current = spacecraft

    // Spacecraft glow
    const spacecraftGlowGeometry = new THREE.SphereGeometry(1800, 16, 16)
    const spacecraftGlowMaterial = new THREE.MeshBasicMaterial({
      color: 0xffd700,
      transparent: true,
      opacity: 0.5,
    })
    const spacecraftGlow = new THREE.Mesh(spacecraftGlowGeometry, spacecraftGlowMaterial)
    spacecraft.add(spacecraftGlow)

    // Add orbital mechanics visualization
    // Low Earth Orbit (LEO) ring
    const leoGeometry = new THREE.RingGeometry(8000, 8500, 64)
    const leoMaterial = new THREE.MeshBasicMaterial({
      color: 0x00ff00,
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide,
    })
    const leoRing = new THREE.Mesh(leoGeometry, leoMaterial)
    leoRing.rotation.x = Math.PI / 2
    scene.add(leoRing)

    // Geostationary orbit ring
    const geoGeometry = new THREE.RingGeometry(42000, 42100, 64)
    const geoMaterial = new THREE.MeshBasicMaterial({
      color: 0x0088ff,
      transparent: true,
      opacity: 0.2,
      side: THREE.DoubleSide,
    })
    const geoRing = new THREE.Mesh(geoGeometry, geoMaterial)
    geoRing.rotation.x = Math.PI / 2
    scene.add(geoRing)

    // Lunar orbit ring
    const lunarOrbitGeometry = new THREE.RingGeometry(59000, 59100, 64)
    const lunarOrbitMaterial = new THREE.MeshBasicMaterial({
      color: 0x888888,
      transparent: true,
      opacity: 0.2,
      side: THREE.DoubleSide,
    })
    const lunarOrbitRing = new THREE.Mesh(lunarOrbitGeometry, lunarOrbitMaterial)
    lunarOrbitRing.position.set(60000, 30000, 10000)
    lunarOrbitRing.rotation.x = Math.PI / 2
    lunarOrbitRing.rotation.z = Math.PI / 6 // Tilt the orbit slightly
    scene.add(lunarOrbitRing)

    // Sample trajectories for demonstration - now in 3D with proper Earth-Moon alignment
    const sampleTrajectories = [
      {
        path: [
          [6371, 0, 0], // Start from Earth surface
          [15000, 7500, 2500],
          [30000, 15000, 5000],
          [45000, 22500, 7500],
          [58363, 30000, 10000], // End at Moon surface
        ],
        color: "lime",
        width: 4,
        is_best: true,
        name: "Optimal Direct Path",
      },
      {
        path: [
          [6371, 0, 0], // Start from Earth surface
          [20000, 10000, 3000],
          [35000, 17500, 5500],
          [50000, 25000, 8000],
          [58363, 30000, 10000], // End at Moon surface
        ],
        color: "orange",
        width: 3,
        is_best: false,
        name: "Safe Alternative",
      },
      {
        path: [
          [6371, 0, 0], // Start from Earth surface
          [25000, 12500, 4000],
          [40000, 20000, 7000],
          [55000, 27500, 9000],
          [58363, 30000, 10000], // End at Moon surface
        ],
        color: "cyan",
        width: 2,
        is_best: false,
        name: "Baseline Route",
      },
    ]

    // Add sample trajectories to scene
    sampleTrajectories.forEach((traj) => {
      const points = traj.path.map((p) => new THREE.Vector3(p[0], p[1], p[2]))
      const geometry = new THREE.BufferGeometry().setFromPoints(points)

      let color
      switch (traj.color) {
        case "lime":
          color = 0x00ff00
          break
        case "orange":
          color = 0xffa500
          break
        case "cyan":
          color = 0x00ffff
          break
        case "deepskyblue":
          color = 0x00bfff
          break
        default:
          color = 0xffffff
      }

      const material = new THREE.LineBasicMaterial({
        color: color,
        linewidth: traj.width || 2,
        opacity: traj.is_best ? 1.0 : 0.7,
        transparent: true,
      })

      const line = new THREE.Line(geometry, material)
      scene.add(line)
      trajectoryLinesRef.current.push(line)
    })

    // Enhanced starfield
    const starsGeometry = new THREE.BufferGeometry()
    const starsCount = 20000
    const positions = new Float32Array(starsCount * 3)
    const colors = new Float32Array(starsCount * 3)

    for (let i = 0; i < starsCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 4000000
      positions[i + 1] = (Math.random() - 0.5) * 4000000
      positions[i + 2] = (Math.random() - 0.5) * 4000000

      const starColor = Math.random()
      if (starColor > 0.95) {
        colors[i] = 1
        colors[i + 1] = 0.5
        colors[i + 2] = 0.5 // Red giant
      } else if (starColor > 0.9) {
        colors[i] = 0.5
        colors[i + 1] = 0.5
        colors[i + 2] = 1 // Blue giant
      } else if (starColor > 0.85) {
        colors[i] = 1
        colors[i + 1] = 1
        colors[i + 2] = 0.5 // Yellow star
      } else {
        colors[i] = 1
        colors[i + 1] = 1
        colors[i + 2] = 1 // White dwarf
      }
    }

    starsGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    starsGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3))
    const starsMaterial = new THREE.PointsMaterial({
      size: 200,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
    })
    const stars = new THREE.Points(starsGeometry, starsMaterial)
    scene.add(stars)

    // Enhanced Mouse Controls
    const handleMouseDown = (event) => {
      controlsRef.current.mouseDown = true
      controlsRef.current.previousMousePosition = {
        x: event.clientX,
        y: event.clientY,
      }
    }

    const handleMouseUp = () => {
      controlsRef.current.mouseDown = false
    }

    const handleMouseMove = (event) => {
      if (controlsRef.current.mouseDown) {
        const deltaMove = {
          x: event.clientX - controlsRef.current.previousMousePosition.x,
          y: event.clientY - controlsRef.current.previousMousePosition.y,
        }

        controlsRef.current.cameraRotation.y += deltaMove.x * 0.01
        controlsRef.current.cameraRotation.x += deltaMove.y * 0.01

        // Clamp vertical rotation
        controlsRef.current.cameraRotation.x = Math.max(
          -Math.PI / 2,
          Math.min(Math.PI / 2, controlsRef.current.cameraRotation.x),
        )

        controlsRef.current.previousMousePosition = {
          x: event.clientX,
          y: event.clientY,
        }
      }
    }

    const handleWheel = (event) => {
      event.preventDefault()
      const zoomSpeed = 0.1
      const zoomFactor = event.deltaY > 0 ? 1 + zoomSpeed : 1 - zoomSpeed
      controlsRef.current.cameraDistance *= zoomFactor
      controlsRef.current.cameraDistance = Math.max(20000, Math.min(500000, controlsRef.current.cameraDistance))
    }

    // Add event listeners
    renderer.domElement.addEventListener("mousedown", handleMouseDown)
    renderer.domElement.addEventListener("mouseup", handleMouseUp)
    renderer.domElement.addEventListener("mousemove", handleMouseMove)
    renderer.domElement.addEventListener("wheel", handleWheel)

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate)

      // Rotate Earth
      earth.rotation.y += 0.003

      // Rotate Moon around its axis
      moon.rotation.y += 0.002

      // Rotate stars slowly
      stars.rotation.x += 0.0002
      stars.rotation.y += 0.0001

      // Update camera position based on controls
      const controls = controlsRef.current
      camera.position.x =
        controls.cameraTarget.x +
        controls.cameraDistance * Math.sin(controls.cameraRotation.y) * Math.cos(controls.cameraRotation.x)
      camera.position.y = controls.cameraTarget.y + controls.cameraDistance * Math.sin(controls.cameraRotation.x)
      camera.position.z =
        controls.cameraTarget.z +
        controls.cameraDistance * Math.cos(controls.cameraRotation.y) * Math.cos(controls.cameraRotation.x)

      camera.lookAt(controls.cameraTarget.x, controls.cameraTarget.y, controls.cameraTarget.z)

      renderer.render(scene, camera)
    }
    animate()

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight)
    }
    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      if (mountRef.current && mountRef.current.contains(renderer.domElement)) {
        renderer.domElement.removeEventListener("mousedown", handleMouseDown)
        renderer.domElement.removeEventListener("mouseup", handleMouseUp)
        renderer.domElement.removeEventListener("mousemove", handleMouseMove)
        renderer.domElement.removeEventListener("wheel", handleWheel)
        mountRef.current.removeChild(renderer.domElement)
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  // Update simulation data
  useEffect(() => {
    if (!sceneRef.current || !sim) return

    const scene = sceneRef.current

    // Clear previous trajectory lines
    trajectoryLinesRef.current.forEach((line) => {
      scene.remove(line)
    })
    trajectoryLinesRef.current = []

    // Clear previous hazards/debris
    const objectsToRemove = []
    scene.traverse((child) => {
      if (child.userData.type === "hazard" || child.userData.type === "debris") {
        objectsToRemove.push(child)
      }
    })
    objectsToRemove.forEach((obj) => scene.remove(obj))

    // Add trajectory lines from simulation
    if (sim.trajectory_options && sim.trajectory_options.length > 0) {
      sim.trajectory_options.forEach((traj) => {
        if (!traj.path || traj.path.length === 0) return

        // Handle both 2D and 3D coordinates
        const points = traj.path.map((p) => {
          if (p.length === 3) {
            return new THREE.Vector3(p[0], p[1], p[2])
          } else {
            return new THREE.Vector3(p[0], p[1], 0)
          }
        })
        const geometry = new THREE.BufferGeometry().setFromPoints(points)

        let color
        switch (traj.color) {
          case "lime":
            color = 0x00ff00
            break
          case "orange":
            color = 0xffa500
            break
          case "cyan":
            color = 0x00ffff
            break
          case "deepskyblue":
            color = 0x00bfff
            break
          case "magenta":
            color = 0xff00ff
            break
          default:
            color = 0xffffff
        }

        const material = new THREE.LineBasicMaterial({
          color: color,
          linewidth: traj.width || 2,
          opacity: traj.is_best ? 1.0 : 0.6,
          transparent: true,
        })

        const line = new THREE.Line(geometry, material)
        scene.add(line)
        trajectoryLinesRef.current.push(line)

        // Add a glow effect for the optimal trajectory
        if (traj.is_best) {
          const glowMaterial = new THREE.LineBasicMaterial({
            color: color,
            linewidth: (traj.width || 2) + 2,
            opacity: 0.3,
            transparent: true,
          })
          const glowLine = new THREE.Line(geometry, glowMaterial)
          scene.add(glowLine)
          trajectoryLinesRef.current.push(glowLine)
        }
      })
    } else {
      // Fallback: Add sample trajectories if no simulation data
      const sampleTrajectories = [
        {
          path: [
            [6371, 0, 0],
            [15000, 7500, 2500],
            [30000, 15000, 5000],
            [45000, 22500, 7500],
            [58363, 30000, 10000],
          ],
          color: "lime",
          width: 4,
          is_best: true,
          name: "Sample Optimal Path",
        },
        {
          path: [
            [6371, 0, 0],
            [20000, 10000, 3000],
            [35000, 17500, 5500],
            [50000, 25000, 8000],
            [58363, 30000, 10000],
          ],
          color: "cyan",
          width: 2,
          is_best: false,
          name: "Sample Alternative",
        }
      ]

      sampleTrajectories.forEach((traj) => {
        const points = traj.path.map((p) => new THREE.Vector3(p[0], p[1], p[2]))
        const geometry = new THREE.BufferGeometry().setFromPoints(points)

        let color = 0x00ff00
        if (traj.color === "cyan") color = 0x00ffff

        const material = new THREE.LineBasicMaterial({
          color: color,
          linewidth: traj.width,
          opacity: traj.is_best ? 1.0 : 0.6,
          transparent: true,
        })

        const line = new THREE.Line(geometry, material)
        scene.add(line)
        trajectoryLinesRef.current.push(line)
      })
    }

    // Add hazards from backend data
    const hazards = sim?.rl?.obstaclesHazards || []
    hazards.forEach((pos) => {
      const geometry = new THREE.SphereGeometry(30, 12, 12)  // Reduced from 2000 to 800
      const material = new THREE.MeshBasicMaterial({
        color: 0xff0000,
        transparent: true,
        opacity: 0.8,
      })
      const hazardMesh = new THREE.Mesh(geometry, material)
      // Handle both 2D and 3D coordinates
      if (pos.length === 3) {
        hazardMesh.position.set(pos[0], pos[1], pos[2])
      } else {
        hazardMesh.position.set(pos[0], pos[1], 0)
      }
      hazardMesh.userData.type = "hazard"
      scene.add(hazardMesh)

      // Add hazard glow (reduced size)
      const glowGeometry = new THREE.SphereGeometry(1200, 8, 8)  // Reduced from 2800 to 1200
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: 0xff0000,
        transparent: true,
        opacity: 0.3,
      })
      const glow = new THREE.Mesh(glowGeometry, glowMaterial)
      hazardMesh.add(glow)
    })

    // Add debris
    const debris = sim?.rl?.obstaclesDebris || []
    debris.forEach((pos) => {
      const geometry = new THREE.BoxGeometry(400, 400, 400)  // Reduced from 1000 to 400
      const material = new THREE.MeshPhongMaterial({
        color: 0xffa500,
        transparent: true,
        opacity: 0.9,
      })
      const debrisObj = new THREE.Mesh(geometry, material)
      // Handle both 2D and 3D coordinates
      if (pos.length === 3) {
        debrisObj.position.set(pos[0], pos[1], pos[2])
      } else {
        debrisObj.position.set(pos[0], pos[1], 0)
      }
      debrisObj.userData.type = "debris"
      debrisObj.rotation.x = Math.random() * Math.PI
      debrisObj.rotation.y = Math.random() * Math.PI
      scene.add(debrisObj)
    })
  }, [sim])

  // Update spacecraft animation
  useEffect(() => {
    if (!spacecraftRef.current) return

    const samplePath = [
      [6371, 0, 0],
      [15000, 7500, 2500],
      [30000, 15000, 5000],
      [45000, 22500, 7500],
      [58363, 30000, 10000],
    ]

    let path = samplePath

    if (sim) {
      // Prioritize RL path from backend
      if (sim?.rl?.path && sim.rl.path.length > 0) {
        path = sim.rl.path
      } else if (sim.trajectory_options) {
        const bestTraj = sim.trajectory_options.find((t) => t.is_best)
        if (bestTraj && bestTraj.path.length > 0) {
          path = bestTraj.path
        } else if (sim.trajectory_options.length > 0) {
          // Fallback to first trajectory if no best is marked
          path = sim.trajectory_options[0].path
        }
      }
    }

    if (path && path.length > 0) {
      const idx = Math.min(animIndex, Math.max(0, path.length - 1))
      const currentPos = path[idx]
      
      // Handle both 2D and 3D coordinates
      if (currentPos.length === 3) {
        spacecraftRef.current.position.set(currentPos[0], currentPos[1], currentPos[2])
      } else {
        spacecraftRef.current.position.set(currentPos[0], currentPos[1], 0)
      }

      // Orient spacecraft toward next waypoint
      if (idx < path.length - 1) {
        const nextPos = path[idx + 1]
        const direction = new THREE.Vector3(
          nextPos[0] - currentPos[0], 
          nextPos[1] - currentPos[1], 
          currentPos.length === 3 ? nextPos[2] - currentPos[2] : 0
        ).normalize()
        
        if (currentPos.length === 3) {
          spacecraftRef.current.lookAt(
            currentPos[0] + direction.x * 1000,
            currentPos[1] + direction.y * 1000,
            currentPos[2] + direction.z * 1000,
          )
        } else {
          spacecraftRef.current.lookAt(
            currentPos[0] + direction.x * 1000,
            currentPos[1] + direction.y * 1000,
            direction.z,
          )
        }
      }

      // Update camera target to follow spacecraft
      if (controlsRef.current) {
        controlsRef.current.cameraTarget = {
          x: currentPos[0] * 0.3,
          y: currentPos[1] * 0.3,
          z: currentPos.length === 3 ? currentPos[2] * 0.3 : 0,
        }
      }
    }
  }, [sim, animIndex])

  return (
    <div className="relative">
      <div
        ref={mountRef}
        style={{ width: "100%", height: "500px", cursor: "grab" }}
        onMouseDown={(e) => (e.target.style.cursor = "grabbing")}
        onMouseUp={(e) => (e.target.style.cursor = "grab")}
      />
      <MissionDetailsPanel sim={sim} />
    </div>
  )
}

function App() {
  const [data, setData] = useState({
    trajectories: [
      {
        path: [
          [8000, 0],
          [15000, 5000],
          [25000, 8000],
          [35000, 12000],
          [45000, 18000],
          [60000, 30000],
        ],
      },
      {
        path: [
          [8000, 0],
          [12000, -3000],
          [20000, -2000],
          [30000, 5000],
          [45000, 20000],
          [60000, 30000],
        ],
      },
      {
        path: [
          [8000, 0],
          [18000, 2000],
          [28000, 6000],
          [38000, 10000],
          [50000, 20000],
          [60000, 30000],
        ],
      },
    ],
    chosen_index: 0,
    scores: [0.95, 0.82, 0.78],
    log: [
      "🚀 ODIN Navigator initialized",
      "📡 Fetching space weather data...",
      "🌍 Earth-Moon trajectory calculated",
      "✅ Optimal path selected",
    ],
  })

  const [sim, setSim] = useState(null)
  const [animIndex, setAnimIndex] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [pauseMsg, setPauseMsg] = useState("")
  const pauseTimerRef = useRef(null)
  const tickTimerRef = useRef(null)

  useEffect(() => {
    setTimeout(() => {
      setData((prevData) => ({
        ...prevData,
        log: [...prevData.log, "🛰 System ready for mission simulation"],
      }))
    }, 1000)
  }, [])

  useEffect(() => {
    return () => {
      clearTimers()
    }
  }, [])

  const clearTimers = () => {
    clearInterval(pauseTimerRef.current)
    clearInterval(tickTimerRef.current)
  }

  const handleSimulate = async () => {
    setIsLoading(true)
    clearTimers()
    try {
      // Call the backend API to get real simulation data
      const response = await fetch('http://localhost:8000/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          windowHours: 72
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const mockSim = await response.json()

      setSim(mockSim)
      setAnimIndex(0)

      const delayHours = mockSim?.rl?.constraints?.suggestedDelayHours || 0
      const secondsPerHour = mockSim?.rl?.timeScale?.simSecondsPerHour ?? 0.5
      if (delayHours > 0) {
        await runPause(`${delayHours}h launch delay due to CME`, Math.max(1, Math.floor(delayHours * secondsPerHour)))
      }
      startTicker(mockSim)
    } catch (error) {
      console.error('Failed to fetch simulation data:', error)
      // Fallback to mock data if API fails
      const fallbackSim = {
        trajectory_options: [
          {
            path: [[6371, 0, 0], [15000, 7500, 2500], [30000, 15000, 5000], [45000, 22500, 7500], [58363, 30000, 10000]],
            color: "lime",
            width: 4,
            is_best: true,
            name: "Optimal RL Path",
          },
          {
            path: [[6371, 0, 0], [20000, 10000, 3000], [35000, 17500, 5500], [50000, 25000, 8000], [58363, 30000, 10000]],
            color: "deepskyblue",
            width: 2,
            is_best: false,
            name: "Baseline A* Path",
          },
          {
            path: [[6371, 0, 5000], [15000, 7500, 7500], [30000, 15000, 10000], [45000, 22500, 12500], [58363, 30000, 15000]],
            color: "cyan",
            width: 2,
            is_best: false,
            name: "High Altitude Route",
          },
          {
            path: [[6371, 0, -2000], [15000, 7500, 1000], [30000, 15000, 4000], [45000, 22500, 7000], [58363, 30000, 10000]],
            color: "orange",
            width: 2,
            is_best: false,
            name: "Low Altitude Direct",
          }
        ],
        plans: [{ name: "Optimal Direct" }],
        bestIndex: 0,
        hazards: [{ pos: [25000, 12000, 5000] }],
        timeframe: { start: "2024-03-15T00:00:00Z", end: "2024-03-18T00:00:00Z" },
        rl: {
          path: [[6371, 0, 0], [15000, 7500, 2500], [30000, 15000, 5000], [45000, 22500, 7500], [58363, 30000, 10000]],
          baseline: [[6371, 0, 0], [20000, 10000, 3000], [35000, 17500, 5500], [50000, 25000, 8000], [58363, 30000, 10000]],
          constraints: { suggestedDelayHours: 0, flarePauseHours: 0, kpMax: 5, dvMultiplierNearEarth: 1.0 },
          timeScale: { simSecondsPerHour: 0.5 },
          pauses: [],
          obstaclesHazards: [[25000, 12000, 5000]],
          obstaclesDebris: [],
          debug: { spacetrackCalled: false, tleCount: 0 },
          ai_analysis: {
            threat_analysis: { threat_level: "LOW" },
            trajectory_evaluation: { recommended_trajectory: "Optimal Direct", confidence_level: "HIGH" },
            ai_recommendations: { immediate_actions: ["API connection failed - using fallback data"] },
            decision_logs: ["⚠️ API connection failed - using fallback simulation data"],
          },
        },
      }
      setSim(fallbackSim)
      setAnimIndex(0)
      startTicker(fallbackSim)
    } finally {
      setIsLoading(false)
    }
  }

  const runPause = (label, seconds) => {
    return new Promise((resolve) => {
      let remaining = Math.max(0, Math.floor(seconds))
      setPauseMsg(`${label} — ${remaining}s`)
      clearInterval(pauseTimerRef.current)
      pauseTimerRef.current = setInterval(() => {
        remaining -= 1
        if (remaining <= 0) {
          clearInterval(pauseTimerRef.current)
          setPauseMsg("")
          resolve()
        } else {
          setPauseMsg(`${label} — ${remaining}s`)
        }
      }, 1000)
    })
  }

  const startTicker = (json) => {
    const pauses = (json?.rl?.pauses || []).slice().sort((a, b) => a.index - b.index)
    const secondsPerHour = json?.rl?.timeScale?.simSecondsPerHour ?? 0.5
    clearInterval(tickTimerRef.current)
    tickTimerRef.current = setInterval(async () => {
      const nextPause = pauses.find((p) => p.index === animIndex)
      if (nextPause && !pauseMsg) {
        clearInterval(tickTimerRef.current)
        await runPause(
          `${nextPause.reason} pause`,
          Math.max(1, nextPause.durationSeconds ?? Math.floor((nextPause.durationHours || 1) * secondsPerHour)),
        )
        startTicker(json)
        return
      }
      setAnimIndex((i) => {
        const pathLength = json?.rl?.path?.length || json?.trajectory_options?.[0]?.path?.length || 6
        const next = i + 1
        return next >= pathLength ? pathLength - 1 : next
      })
    }, 1000)
  }

  const bestTrajectory = useMemo(() => {
    if (!sim?.trajectory_options) return null
    return sim.trajectory_options.find((t) => t.is_best) || sim.trajectory_options[0]
  }, [sim])

  const aiAnalysis = sim?.rl?.ai_analysis
  const decisionLogs = aiAnalysis?.decision_logs || data.log

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      <StarfieldBackground />

      <div className="relative z-10 p-6">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            🚀 ODIN Space Navigator
          </h1>
          <p className="text-gray-300 text-lg">Advanced AI-Powered Mission Planning & Trajectory Optimization</p>
        </header>

        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* Mission Status */}
            <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">Mission Status</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-300">Status:</span>
                  <span className="text-green-400 font-semibold">
                    {isLoading ? "Simulating..." : sim ? "Active" : "Ready"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Trajectory:</span>
                  <span className="text-blue-400">{bestTrajectory?.name || "Optimal Direct"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Progress:</span>
                  <span className="text-yellow-400">
                    {sim ? `${Math.round((animIndex / (bestTrajectory?.path?.length || 6)) * 100)}%` : "0%"}
                  </span>
                </div>
                {aiAnalysis?.threat_analysis && (
                  <div className="flex justify-between">
                    <span className="text-gray-300">Threat Level:</span>
                    <span
                      className={`font-semibold ${
                        aiAnalysis.threat_analysis.threat_level === "LOW"
                          ? "text-green-400"
                          : aiAnalysis.threat_analysis.threat_level === "MODERATE"
                            ? "text-yellow-400"
                            : "text-red-400"
                      }`}
                    >
                      {aiAnalysis.threat_analysis.threat_level}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Trajectory Options */}
            <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 text-purple-400">Trajectory Options</h3>
              <div className="space-y-2">
                {(
                  sim?.trajectory_options || [
                    { name: "Optimal Direct", color: "lime", is_best: true },
                    { name: "Safe Alternative", color: "orange", is_best: false },
                    { name: "Baseline", color: "cyan", is_best: false },
                  ]
                ).map((traj, i) => (
                  <div key={i} className="flex items-center justify-between p-2 rounded bg-gray-800/50">
                    <div className="flex items-center space-x-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{
                          backgroundColor:
                            traj.color === "lime" ? "#00ff00" : traj.color === "orange" ? "#ffa500" : "#00ffff",
                        }}
                      />
                      <span className="text-sm">{traj.name}</span>
                    </div>
                    {traj.is_best && <span className="text-xs bg-green-600 px-2 py-1 rounded">SELECTED</span>}
                  </div>
                ))}
              </div>
            </div>

            {/* AI Recommendations */}
            <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 text-green-400">AI Analysis</h3>
              {aiAnalysis?.ai_recommendations?.immediate_actions ? (
                <div className="space-y-2">
                  {aiAnalysis.ai_recommendations.immediate_actions.map((action, i) => (
                    <div key={i} className="text-sm text-gray-300 flex items-start space-x-2">
                      <span className="text-green-400 mt-1">•</span>
                      <span>{action}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-gray-400 text-sm">AI analysis will appear here during simulation</div>
              )}
              {aiAnalysis?.trajectory_evaluation && (
                <div className="mt-4 p-3 bg-gray-800/50 rounded">
                  <div className="text-sm">
                    <span className="text-gray-400">Recommended: </span>
                    <span className="text-blue-400">{aiAnalysis.trajectory_evaluation.recommended_trajectory}</span>
                  </div>
                  <div className="text-sm mt-1">
                    <span className="text-gray-400">Confidence: </span>
                    <span className="text-green-400">{aiAnalysis.trajectory_evaluation.confidence_level}</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 3D Visualization */}
          <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700 mb-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-semibold text-cyan-400">3D Mission Visualization</h3>
              <div className="text-sm text-gray-400">Mouse: Rotate | Wheel: Zoom | Drag: Pan</div>
            </div>
            <ThreeJSVisualization sim={sim} animIndex={animIndex} />
            {pauseMsg && (
              <div className="mt-4 p-3 bg-yellow-900/50 border border-yellow-600 rounded text-center">
                <span className="text-yellow-300 font-semibold">{pauseMsg}</span>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="mb-6">
            <Controls onSimulate={handleSimulate} isLoading={isLoading} />
          </div>

          {/* Mission Logs */}
          <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h3 className="text-xl font-semibold mb-4 text-orange-400">Mission Logs</h3>
            <div className="bg-black/50 rounded p-4 font-mono text-sm max-h-64 overflow-y-auto">
              <Logs log={decisionLogs} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App