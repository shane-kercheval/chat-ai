export class SidebarManager {
    constructor() {
        this.elements = {
            sidebar: document.getElementById('sidebar'),
            resizeHandle: document.getElementById('resize-handle'),
            tabContents: document.querySelectorAll('.tab-content'),
            sidebarIcons: document.querySelectorAll('.sidebar-icon'),
            toggleButton: document.getElementById('toggle-sidebar')
        };
        
        this.lastWidth = 400; // Default width
        this.initialize();
        this.initializeResizing();
        this.initializeToggle();
    }

    initialize() {
        // Hide all tab contents initially
        this.elements.tabContents.forEach(content => {
            content.style.display = 'none';
        });

        // Show models content by default
        const modelsContent = document.getElementById('models-content');
        if (modelsContent) {
            modelsContent.style.display = 'block';
        }

        this.elements.sidebarIcons.forEach((icon) => {
            icon.addEventListener('click', () => {
                this.elements.sidebarIcons.forEach(i => i.classList.remove('active'));
                icon.classList.add('active');

                this.elements.tabContents.forEach(content => {
                    content.style.display = 'none';
                });

                const tabName = icon.getAttribute('data-tab');
                const selectedTab = document.getElementById(`${tabName}-content`);
                if (selectedTab) {
                    selectedTab.style.display = 'block';
                }
            });
        });
    }
    
    initializeResizing() {
        let isResizing = false;
        let startX;
        let startWidth;

        this.elements.resizeHandle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.clientX;
            startWidth = this.elements.sidebar.offsetWidth;
            
            this.elements.sidebar.classList.add('resizing');
        });

        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;

            const width = startWidth + (e.clientX - startX);
            // Set minimum and maximum width constraints
            if (width >= 48 && width <= 800) {
                this.elements.sidebar.style.width = `${width}px`;
                this.lastWidth = width; // Save the last manually set width
            }
        });

        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                this.elements.sidebar.classList.remove('resizing');
            }
        });
    }

    toggleSidebar() {
        const isExpanded = this.elements.sidebar.classList.contains('expanded');
        if (isExpanded) {
            // Save current width before collapsing if it's not already saved
            if (this.elements.sidebar.style.width) {
                this.lastWidth = this.elements.sidebar.offsetWidth;
            }
            this.elements.sidebar.style.width = '';
            this.elements.sidebar.classList.remove('expanded');
            this.elements.sidebar.classList.add('collapsed');
            this.elements.toggleButton.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 18l6-6-6-6"></path>
                </svg>
            `;
        } else {
            this.elements.sidebar.classList.remove('collapsed');
            this.elements.sidebar.classList.add('expanded');
            this.elements.sidebar.style.width = `${this.lastWidth}px`;
            this.elements.toggleButton.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M15 18l-6-6 6-6"></path>
                </svg>
            `;
        }
    }

    initializeToggle() {
        this.elements.toggleButton.addEventListener('click', () => {
            this.toggleSidebar();
        });
    }
}
